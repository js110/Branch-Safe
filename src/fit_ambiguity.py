"""Construct a finite R-BSP ambiguity set from development/validation mobility data.

GeoLife contains trajectories but no real task-delivery/willingness/deadline logs.
Therefore mobility and prior uncertainty are data-derived here, while alpha is an
explicit externally supplied protocol parameter.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .geolife import effective_geolife_splits,state_path_fingerprint


def state_exposure(states,side):
    states=np.asarray(states,dtype=int)
    x=states//side;y=states%side
    degree=4-(x==0).astype(int)-(x==side-1).astype(int)-(y==0).astype(int)-(y==side-1).astype(int)
    return degree/4.0


def fit_move_coarsened(paths,side=8):
    """MLE for P(cell changes | state)=v*deg(state)/4.

    Destination direction/distance is deliberately discarded. This matches the
    paper's reflecting-walk approximation while avoiding zero likelihood for
    observed multi-cell jumps.
    """
    paths=np.asarray(paths,dtype=int)
    src=paths[:,:-1].ravel();changed=(paths[:,1:]!=paths[:,:-1]).ravel().astype(float)
    exposure=state_exposure(src,side)
    nchange=float(changed.sum())
    if nchange==0:return 0.0
    stay=changed==0
    def deriv(v):
        return nchange/max(v,1e-15)-float(np.sum(exposure[stay]/np.maximum(1-v*exposure[stay],1e-15)))
    hi=1-1e-12
    if deriv(hi)>=0:return 1.0
    lo=1e-12
    for _ in range(80):
        mid=(lo+hi)/2
        if deriv(mid)>0:lo=mid
        else:hi=mid
    return float((lo+hi)/2)


def state_path_groups(paths):
    groups={}
    for i,path in enumerate(np.asarray(paths)):
        groups.setdefault(state_path_fingerprint(path),[]).append(i)
    return list(groups.values())

def group_bootstrap(paths,side,reps,seed):
    rng=np.random.default_rng(seed);paths=np.asarray(paths)
    groups=state_path_groups(paths)
    out=np.empty(reps,float)
    for r in range(reps):
        picked=rng.integers(len(groups),size=len(groups))
        idx=np.concatenate([np.asarray(groups[j],dtype=int) for j in picked])
        out[r]=fit_move_coarsened(paths[idx],side)
    return out


def smoothed_prior(paths,side,pseudocount=.5):
    counts=np.bincount(np.asarray(paths,dtype=int).ravel(),minlength=side*side).astype(float)
    counts+=float(pseudocount)
    return (counts/counts.sum()).tolist()


def per_path_loglik(paths,side,move,prior):
    paths=np.asarray(paths,dtype=int);prior=np.asarray(prior,float)
    out=np.zeros(len(paths),float)
    for j,path in enumerate(paths):
        src=path[:-1];changed=(path[1:]!=path[:-1])
        e=state_exposure(src,side);p=np.clip(float(move)*e,1e-12,1-1e-12)
        out[j]=np.log(max(prior[int(path[0])],1e-300))+np.where(changed,np.log(p),np.log1p(-p)).sum()
    return out


def validation_bootstrap_support(paths,candidates,side,reps,seed,mass=.95):
    scores=np.column_stack([per_path_loglik(paths,side,c['move'],c['prior']) for c in candidates])
    groups=state_path_groups(paths)
    rng=np.random.default_rng(seed);wins=np.zeros(len(candidates),int)
    for _ in range(reps):
        picked=rng.integers(len(groups),size=len(groups))
        idx=np.concatenate([np.asarray(groups[j],dtype=int) for j in picked])
        total=scores[idx].sum(axis=0);wins[int(np.argmax(total))]+=1
    freq=wins/reps;order=np.argsort(-freq,kind='stable');keep=[];cum=0.
    for i in order:
        if freq[i]<=0 and keep:break
        keep.append(int(i));cum+=float(freq[i])
        if cum>=mass-1e-12:break
    return freq,keep,float(cum)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--development',default='data/geolife_development.npz')
    ap.add_argument('--validation',default='data/geolife_validation.npz')
    ap.add_argument('--output',default='results/data_driven_ambiguity/models.json')
    ap.add_argument('--side',type=int,default=8)
    ap.add_argument('--alpha',type=float,default=.648)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--confidence',type=float,default=.95)
    ap.add_argument('--selection-mass',type=float,default=.95)
    ap.add_argument('--seed',type=int,default=270926)
    a=ap.parse_args()
    if not (0<a.alpha<=1):raise ValueError('alpha must lie in (0,1]')
    if not (0<a.selection_mass<=1):raise ValueError('selection mass must lie in (0,1]')
    splits,_=effective_geolife_splits({'development':a.development,'validation':a.validation})
    dev=splits['development'];val=splits['validation']
    dp=np.asarray(dev['paths']);vp=np.asarray(val['paths'])
    dev_hat=fit_move_coarsened(dp,a.side);val_hat=fit_move_coarsened(vp,a.side)
    pooled_hat=fit_move_coarsened(np.concatenate([dp,vp]),a.side)
    db=group_bootstrap(dp,a.side,a.bootstrap,a.seed)
    vb=group_bootstrap(vp,a.side,a.bootstrap,a.seed+1)
    tail=(1-a.confidence)/2
    dci=np.quantile(db,[tail,1-tail]);vci=np.quantile(vb,[tail,1-tail])
    move_q=np.quantile(db,[.025,.25,.5,.75,.975])
    moves=[]
    for x in list(move_q)+[dev_hat]:
        if not any(abs(float(x)-y)<1e-10 for y in moves):moves.append(float(x))
    pop_prior=smoothed_prior(dp,a.side,.5)
    uniform=(np.ones(a.side*a.side)/(a.side*a.side)).tolist()
    candidates=[]
    for move in sorted(moves):
        candidates.append(dict(move=move,alpha=a.alpha,prior=uniform,prior_type='uniform'))
        candidates.append(dict(move=move,alpha=a.alpha,prior=pop_prior,prior_type='population_smoothed'))
    freq,keep,cum=validation_bootstrap_support(vp,candidates,a.side,a.bootstrap,a.seed+2,a.selection_mass)
    models=[]
    candidate_records=[]
    for i,m in enumerate(candidates):
        rec=dict(index=i,move=m['move'],prior_type=m['prior_type'],validation_selection_frequency=float(freq[i]),selected=i in keep)
        candidate_records.append(rec)
        if i in keep:models.append(m)
    out=dict(
        schema='data-driven-rbsp-v3',
        construction='state-path-group development bootstrap candidates; state-path-group validation bootstrap predictive-likelihood support set',
        side=a.side,alpha=a.alpha,
        alpha_provenance='externally specified protocol parameter; GeoLife has trajectories but no real task availability logs',
        fit_model='coarsened reflecting-walk change likelihood P(change|state)=v*degree(state)/4; destination direction and jump distance ignored',
        development=dict(users=int(len(dp)),windows=int(len(dp)),unique_model_input_paths=int(len(state_path_groups(dp))),estimate=dev_hat,ci=dci.tolist(),candidate_quantiles=[.025,.25,.5,.75,.975]),
        validation=dict(users=int(len(vp)),windows=int(len(vp)),unique_model_input_paths=int(len(state_path_groups(vp))),estimate=val_hat,ci=vci.tolist()),
        pooled=dict(users=int(len(dp)+len(vp)),windows=int(len(dp)+len(vp)),unique_model_input_paths=int(len(state_path_groups(np.concatenate([dp,vp])))),estimate=pooled_hat),
        confidence=a.confidence,bootstrap_replicates=a.bootstrap,bootstrap_seed=a.seed,
        selection_mass_target=a.selection_mass,selection_mass_achieved=cum,
        move_candidates=sorted(moves),candidate_models=candidate_records,
        priors=['uniform','population_smoothed_development_only'],prior_pseudocount=.5,
        population_prior=pop_prior,models=models,
        test_data_used=False)
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(development=out['development'],validation=out['validation'],
                          selected_models=[candidate_records[i] for i in keep],
                          selection_mass_achieved=cum),indent=2))


if __name__=='__main__':main()
