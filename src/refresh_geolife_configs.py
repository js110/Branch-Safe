"""Refresh GeoLife-fitted movement parameters in tracked replay configs.

The raw NPZ caches intentionally remain immutable audit inputs.  Current replay
configs use the effective exact-coordinate-decontaminated development split.
"""
import json
from pathlib import Path
from .geolife import effective_development_change_probability

CONFIGS=(
    Path("configs/final.json"),
    Path("configs/kl_baseline.json"),
    Path("configs/recent_baselines.json"),
    Path("configs/recent_privic_thinning.json"),
)

def refresh():
    move=effective_development_change_probability()
    changed={}
    for path in CONFIGS:
        cfg=json.loads(path.read_text())
        n=0
        for condition in cfg["conditions"]:
            if condition.get("scenario")=="geolife":
                condition["move_model"]=move
                n+=1
        path.write_text(json.dumps(cfg,indent=2)+"\n")
        changed[str(path)]=n
    return {"effective_development_change_probability":move,"conditions_updated":changed}

if __name__=="__main__":
    print(json.dumps(refresh(),indent=2))
