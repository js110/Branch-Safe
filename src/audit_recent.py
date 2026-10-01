"""Reconstruct recent channels from public history; audit native fit diagnostics."""
import gzip,hashlib,json
import argparse
from pathlib import Path
import numpy as np
from .model import Task,predict
from .recent_baselines import pml_gates

def training_audit():
    manifest=json.loads(Path('results/privic_training/manifest.json').read_text());diagnostics=[];limit_hits=[]
    for row in manifest:
        path=Path(row['channel_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        z=np.load(path);p=z['estimate'];channels=z['round_channels'];counts=z['round_counts'];n=counts.sum()
        # For concave mean log likelihood f(theta), f(theta*)-f(theta)
        # <= max_i grad_i f(theta) - theta dot grad f(theta) = max_i grad_i - 1.
        grad=sum(c@np.divide(cnt,p@c,out=np.zeros_like(p),where=p@c>0) for c,cnt in zip(channels,counts))/n
        gap=max(0,float(grad.max()-np.dot(p,grad)))
        diagnostics.append(dict(scenario=row['scenario'],beta=row['beta'],gibu_mean_log_likelihood_gap_upper_bound=gap,
                                converged=row['gibu']['converged'],final_ba_converged=row['final_ba']['converged']))
        for label,info in [('final_ba',row['final_ba']),('gibu',row['gibu'])]+[(f'round_{i}_{k}',s[k]) for i,s in enumerate(row['rounds']) for k in ['ba','ibu']]:
            if not info['converged']:limit_hits.append(dict(scenario=row['scenario'],beta=row['beta'],stage=label,residual=info['residual']))
    out=Path('results/privic_training')
    (out/'optimality_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (out/'convergence_audit.json').write_text(json.dumps(dict(limit_hits=limit_hits,wall_seconds=sum(r['wall_seconds'] for r in manifest)),indent=2))
    return dict(channels=len(manifest),limit_hits=len(limit_hits),final_ba_converged=sum(d['final_ba_converged'] for d in diagnostics),
                final_gibu_limit_hits=sum(not d['converged'] for d in diagnostics),largest_mean_log_likelihood_gap=max(d['gibu_mean_log_likelihood_gap_upper_bound'] for d in diagnostics))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='results/recent_baselines');args=ap.parse_args()
    report=training_audit();run=Path(args.run);cfg=json.loads((run/'config.json').read_text());events=0;cases=0;max_error=0
    for cond in cfg['conditions']:
        side=cond['side'];alpha=cond['delivery']*cond['willing']*cond['on_time']
        c=np.load(cond['channel_file'])['channel'] if cond['method']=='privic' else None
        references={}
        with gzip.open(Path('results/final')/(cond['scenario']+'_adaptive_none_1.jsonl.gz'),'rt') as f:
            for line in f:
                r=json.loads(line);references[(r['seed'],r['user'])]=r['events']
        with gzip.open(run/(cond['id']+'.jsonl.gz'),'rt') as f:
            for line in f:
                r=json.loads(line);b=np.full(side*side,1/(side*side));cases+=1;ref=references[(r['seed'],r['user'])]
                for e in r['events']:
                    t=e['slot'];events+=1
                    if t:b=predict(b,side,cond['move_model'])
                    mask=Task('',t,t,tuple(e['rectangle'])).mask(side)
                    q=cond.get('gate_scale',1.0)*(c@mask) if c is not None else pml_gates(b,mask,alpha,cond['param'])[0]
                    np.testing.assert_allclose(q,e['q'],rtol=1e-9,atol=1e-11)
                    ell=alpha*mask*q
                    if c is None:
                        for branch in [ell,1-ell]:
                            marginal=b@branch
                            if marginal>1e-12:assert max(branch[b>0])/marginal<=np.exp(cond['param'])+1e-7
                    posterior=b*(ell if e['reported'] else 1-ell);posterior/=posterior.sum()
                    err=float(np.max(np.abs(posterior-e['belief'])));max_error=max(max_error,err)
                    assert err<1e-9
                    assert e['legitimate']==ref[t]['legitimate'] and e['available']==ref[t]['available']
                    if e['legitimate']:assert e['rectangle']==ref[t]['rectangle']
                    b=posterior
    report.update(conditions=len(cfg['conditions']),trajectories=cases,events=events,max_posterior_error=max_error,
                  shared_normal_tasks_and_availability=True,public_channel_reconstructed=True,issues=[])
    (run/'channel_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
