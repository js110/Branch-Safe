"""GeoLife split loading with model-input-level decontamination.

The committed NPZ files are immutable deterministic caches of the original
user-ID split candidates. Effective development/validation/test splits are
formed in that order. Exact GPS-coordinate duplicates are tracked for
provenance, while holdout isolation is enforced on the complete 48-slot
coarse state path actually consumed by the fitted/replay models.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

SPLIT_ORDER=("development","validation","test")

def _sha256_array(values,dtype):
    a=np.asarray(values,dtype=dtype)
    a=np.ascontiguousarray(a.astype(dtype,copy=False))
    return hashlib.sha256(a.tobytes()).hexdigest()

def coordinate_fingerprint(coords):
    """Stable SHA-256 fingerprint for an exact coordinate window."""
    return _sha256_array(coords,">f8")

def state_path_fingerprint(path):
    """Stable SHA-256 fingerprint for the full discrete model-input path."""
    return _sha256_array(path,">i8")

def _read_npz(path):
    p=Path(path)
    with np.load(p) as data:
        return {k:np.array(data[k],copy=True) for k in ("users","paths","coords")}

def effective_geolife_splits(paths=None):
    """Return effective splits with no repeated model-input path across splits.

    Split precedence is development, then validation, then test. Each complete
    discrete state-path group is assigned to the earliest split in which it
    occurs. All members of that group in the owning split are retained, while
    members in later splits are excluded. Thus model-input groups cannot cross
    effective splits, without erasing repeated observations inside one split.

    Exact coordinate duplicates are also recorded separately for provenance.
    """
    if paths is None:
        paths={s:Path("data")/f"geolife_{s}.npz" for s in SPLIT_ORDER}
    else:
        paths={s:Path(p) for s,p in paths.items()}
    raw={s:_read_npz(paths[s]) for s in SPLIT_ORDER if s in paths}
    owner_by_state={}
    first_coordinate={}
    out={}
    dropped=[]
    coordinate_duplicates=[]
    for split in SPLIT_ORDER:
        if split not in raw:continue
        data=raw[split];keep=[]
        for i,(user,path,coords) in enumerate(zip(data["users"],data["paths"],data["coords"])):
            user=int(user)
            sfp=state_path_fingerprint(path)
            cfp=coordinate_fingerprint(coords)
            if cfp in first_coordinate:
                prior=first_coordinate[cfp]
                coordinate_duplicates.append(dict(
                    split=split,user=user,coordinate_fingerprint=cfp,
                    duplicate_of_split=prior["split"],
                    duplicate_of_user=prior["user"]))
            else:
                first_coordinate[cfp]=dict(split=split,user=user)

            owner=owner_by_state.get(sfp)
            if owner is None:
                owner_by_state[sfp]={"split":split,"user":user}
            elif owner["split"]!=split:
                dropped.append(dict(
                    split=split,user=user,state_path_fingerprint=sfp,
                    assigned_split=owner["split"],
                    representative_user=owner["user"],
                    coordinate_fingerprint=cfp,
                    reason="model-input group assigned to earlier split"))
                continue
            keep.append(i)
        idx=np.asarray(keep,dtype=int)
        out[split]={k:v[idx] for k,v in data.items()}
    return out,dropped

def effective_state_group_by_user(split="test",paths=None):
    """Map effective-split user IDs to complete model-input state-path groups."""
    splits,_=effective_geolife_splits(paths)
    data=splits[split]
    return {int(user):state_path_fingerprint(path) for user,path in zip(data["users"],data["paths"])}

def load_effective_geolife(path):
    """Load one split, enforcing model-input group isolation against earlier splits."""
    p=Path(path)
    split=None
    for s in SPLIT_ORDER:
        if p.name==f"geolife_{s}.npz":
            split=s;break
    if split is None:
        return _read_npz(p)

    upto=SPLIT_ORDER.index(split)
    paths={}
    for s in SPLIT_ORDER[:upto+1]:
        q=p.with_name(f"geolife_{s}.npz")
        if q.exists():paths[s]=q
    if split not in paths:
        paths[split]=p
    splits,_=effective_geolife_splits(paths)
    return splits[split]

def effective_development_change_probability(paths=None):
    """Mean per-window coarse-state change rate on effective development windows."""
    splits,_=effective_geolife_splits(paths)
    dev=np.asarray(splits["development"]["paths"])
    if len(dev)==0:
        raise ValueError("effective development split is empty")
    return float(np.mean([np.mean(np.diff(path)!=0) for path in dev]))

def decontamination_audit(paths=None):
    """Machine-readable audit of raw caches versus effective model-input groups."""
    if paths is None:
        paths={s:Path("data")/f"geolife_{s}.npz" for s in SPLIT_ORDER}
    else:
        paths={s:Path(p) for s,p in paths.items()}
    raw={s:_read_npz(paths[s]) for s in SPLIT_ORDER if s in paths}
    effective,dropped=effective_geolife_splits(paths)

    coord_seen={}
    coord_dups=[]
    state_groups={}
    for split in SPLIT_ORDER:
        if split not in raw:continue
        for user,path,coords in zip(raw[split]["users"],raw[split]["paths"],raw[split]["coords"]):
            user=int(user)
            cfp=coordinate_fingerprint(coords)
            sfp=state_path_fingerprint(path)
            if cfp in coord_seen:
                prior=coord_seen[cfp]
                coord_dups.append(dict(
                    split=split,user=user,coordinate_fingerprint=cfp,
                    duplicate_of_split=prior["split"],
                    duplicate_of_user=prior["user"]))
            else:
                coord_seen[cfp]=dict(split=split,user=user)
            state_groups.setdefault(sfp,[]).append(dict(split=split,user=user,coordinate_fingerprint=cfp))

    cross_split_groups=[
        dict(state_path_fingerprint=fp,members=members)
        for fp,members in state_groups.items()
        if len({m["split"] for m in members})>1
    ]
    return {
        "holdout_rule":"Group by SHA-256 of the complete discrete state path consumed by the model and assign each group to its earliest development -> validation -> test split. Retain all group members in the owning split and exclude later-split members. Exact GPS-coordinate fingerprints are retained as a separate provenance audit.",
        "raw_counts":{s:int(len(raw[s]["users"])) for s in raw},
        "effective_counts":{s:int(len(effective[s]["users"])) for s in effective},
        "unique_model_input_paths":int(len(state_groups)),
        "cross_split_model_input_groups":cross_split_groups,
        "dropped_later_split_windows":dropped,
        "exact_coordinate_duplicates":coord_dups,
        "development_change_probability":effective_development_change_probability(paths),
    }

if __name__=="__main__":
    print(json.dumps(decontamination_audit(),indent=2))
