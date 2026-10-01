"""Analyze the frozen finite-model R-BSP stress test.

The analysis is exploratory: the ambiguity grid was chosen after the nominal BSP
study exposed model-mismatch failures. Bootstrap units are synthetic seeds.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np

METRICS=['hit','peak','logloss','utility','raw_completion','gate_us_p50',
         'attacker_local_violation_rate','attacker_absolute_violation_rate',
         'client_violation_rate']

def load_rows(run):
    rows=list(csv.DictReader((run/'rows.csv').open()))
    numeric=set(rows[0])-{'condition','scenario','method','attack','user'}
    for r in rows:
        for k in numeric:r[k]=float(r[k])
    return rows

def aggregate(rows,slots):
    n=len(rows)
    opp=sum(r['opportunities'] for r in rows);complete=sum(r['complete'] for r in rows)
    return {
        'hit':float(np.mean([r['hit'] for r in rows])),
        'peak':float(np.mean([r['peak'] for r in rows])),
        'logloss':float(np.mean([r['logloss'] for r in rows])),
        'utility':float(complete/opp) if opp else float('nan'),
        'raw_completion':float(np.mean([r['raw_completion'] for r in rows])),
        'gate_us_p50':float(np.mean([r['gate_us_p50'] for r in rows])),
        'attacker_local_violation_rate':float(sum(r['attacker_local_cap_violations'] for r in rows)/(n*slots)),
        'attacker_absolute_violation_rate':float(sum(r['attacker_absolute_cap_violations'] for r in rows)/(n*slots)),
        'client_violation_rate':float(sum(r['cap_violations'] for r in rows)/(n*slots)),
    }

def summarize(rows,cfg,nboot=1000):
    slots=int(cfg['slots']);by_id={c['id']:c for c in cfg['conditions']};out=[]
    for cid in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==cid];cond=by_id[cid]
        seeds=sorted(set(int(r['seed']) for r in group))
        by_seed={s:[r for r in group if int(r['seed'])==s] for s in seeds}
        central=aggregate(group,slots);rng=np.random.default_rng(92626);draws={k:[] for k in METRICS}
        for _ in range(nboot):
            sampled=rng.choice(seeds,size=len(seeds),replace=True)
            sample=[r for s in sampled for r in by_seed[int(s)]]
            a=aggregate(sample,slots)
            for k in METRICS:draws[k].append(a[k])
        d={'condition':cid,'mismatch':cond['mismatch'],'method':cond['method'],'param':float(cond['param']),
           'n_seeds':len(seeds),'n_trajectories':len(group),'robust_model_count':int(group[0]['robust_model_count'])}
        for k in METRICS:
            vals=np.asarray(draws[k],float);vals=vals[np.isfinite(vals)]
            d[k]=central[k]
            d[k+'_low']=float(np.quantile(vals,.025)) if len(vals) else float('nan')
            d[k+'_high']=float(np.quantile(vals,.975)) if len(vals) else float('nan')
        out.append(d)
    return out

def paired(summary):
    out=[]
    for mismatch in sorted(set(r['mismatch'] for r in summary)):
        for param in sorted(set(r['param'] for r in summary if r['mismatch']==mismatch)):
            b=next(r for r in summary if r['mismatch']==mismatch and r['param']==param and r['method']=='bsp')
            r=next(r for r in summary if r['mismatch']==mismatch and r['param']==param and r['method']=='rbsp')
            out.append({'mismatch':mismatch,'param':param,
                        'bsp_hit':b['hit'],'rbsp_hit':r['hit'],'hit_delta_rbsp_minus_bsp':r['hit']-b['hit'],
                        'bsp_utility':b['utility'],'rbsp_utility':r['utility'],'utility_delta_rbsp_minus_bsp':r['utility']-b['utility'],
                        'bsp_local_violation_rate':b['attacker_local_violation_rate'],
                        'rbsp_local_violation_rate':r['attacker_local_violation_rate'],
                        'bsp_client_violation_rate':b['client_violation_rate'],
                        'rbsp_client_violation_rate':r['client_violation_rate']})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='results/robust_informed');a=ap.parse_args()
    run=Path(a.run);cfg=json.loads((run/'config.json').read_text());rows=load_rows(run)
    summary=summarize(rows,cfg);out=run/'analysis';out.mkdir(exist_ok=True)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    pairs=paired(summary);(out/'paired.json').write_text(json.dumps(pairs,indent=2))
    (out/'provenance.json').write_text(json.dumps({
        'status':'exploratory stress test, not confirmatory',
        'bootstrap':1000,'unit':'synthetic seed with four users',
        'ambiguity_grid':cfg.get('robust_model_grid'),
        'primary_security_endpoint':'attacker_local_violation_rate',
        'note':'R-BSP was motivated by previously observed model-mismatch failures; results must not be described as pre-specified confirmation.'
    },indent=2))
    print('Analyzed',len(rows),'trajectories and',len(summary),'conditions')
if __name__=='__main__':main()
