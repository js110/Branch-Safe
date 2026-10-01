"""Analyze targeted TMC reviewer stress tests with seed-cluster bootstrap."""
import csv,json,hashlib
from pathlib import Path
import numpy as np

RUN=Path("results/tmc_reviewer_stress")
CFG=Path("configs/tmc_reviewer_stress.json")

def load_rows():
    rows=list(csv.DictReader((RUN/"rows.csv").open()))
    for r in rows:
        for k,v in list(r.items()):
            if k in ("condition","scenario","method","attack","user"):continue
            r[k]=float("nan") if v in ("","None") else float(v)
    return rows

def aggregate(rows,slots):
    n=len(rows)
    hit=float(np.mean([r["hit"] for r in rows]))
    peak=float(np.mean([r["peak"] for r in rows]))
    local=float(sum(r["attacker_local_cap_violations"] for r in rows)/(n*slots))
    effective=float(sum(r["attacker_effective_cap_violations"] for r in rows)/(n*slots))
    denom=sum(r["opportunities"] for r in rows)
    utility=(sum(r["complete"] for r in rows)/denom) if denom else float("nan")
    return dict(hit=hit,peak=peak,local_violation=local,effective_violation=effective,utility=utility)

def summarize(rows,cfg,nboot=1000):
    slots=int(cfg["slots"]);out=[]
    rng=np.random.default_rng(93026)
    for cond in cfg["conditions"]:
        cid=cond["id"];a=[r for r in rows if r["condition"]==cid]
        seeds=sorted({int(r["seed"]) for r in a})
        by={s:[r for r in a if int(r["seed"])==s] for s in seeds}
        central=aggregate(a,slots)
        draws=[]
        for _ in range(nboot):
            pick=rng.choice(seeds,size=len(seeds),replace=True)
            sample=[r for s in pick for r in by[int(s)]]
            draws.append(aggregate(sample,slots))
        row=dict(condition=cid,stress=cid.split("_")[0],method=cond["method"],attack=cond["attack"],
                 n_seeds=len(seeds),n_trajectories=len(a),**central)
        for k in ["hit","peak","local_violation","effective_violation","utility"]:
            vals=np.asarray([d[k] for d in draws],float);vals=vals[np.isfinite(vals)]
            row[k+"_low"]=float(np.quantile(vals,.025)) if len(vals) else None
            row[k+"_high"]=float(np.quantile(vals,.975)) if len(vals) else None
        out.append(row)
    return out

def pct(v):
    return "--" if v is None or not np.isfinite(v) else f"{100*v:.2f}"

def table(rows):
    labels={"locavail":"Spatial availability","nonstationary":"Nonstationary mobility","mismatch":"Stronger probe"}
    lines=[r"\begin{tabular}{lllrll}",r"\toprule",
           r"Stress & Filter & Probe & Local viol. (\%) & MAP hit (\%) & Opp. retention (\%) \\",
           r"\midrule"]
    for r in rows:
        attack="2-step" if r["attack"]=="lookahead2" else "1-step"
        lines.append(f"{labels[r['stress']]} & {r['method'].upper()} & {attack} & {pct(r['local_violation'])} & {pct(r['hit'])} & {pct(r['utility'])} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    return "\n".join(lines)+"\n"

def main():
    cfg=json.loads(CFG.read_text());rows=load_rows();summ=summarize(rows,cfg)
    out=RUN/"analysis";out.mkdir(parents=True,exist_ok=True)
    fields=list(summ[0])
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summ)
    (out/"summary.json").write_text(json.dumps(summ,indent=2)+"\n")
    prov=dict(schema="tmc-reviewer-stress-v1",bootstrap_replicates=1000,cluster="synthetic seed",
              config_sha256=hashlib.sha256(CFG.read_bytes()).hexdigest(),
              rows_sha256=hashlib.sha256((RUN/"rows.csv").read_bytes()).hexdigest(),
              interpretation="post-review sensitivity/stress evidence; not production-service or confirmatory evidence")
    (out/"provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
    Path("paper/reviewer_stress_table.tex").write_text(table(summ))

if __name__=="__main__":main()
