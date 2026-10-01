"""Audit saved numerical evidence without rerunning or changing experiments."""
import csv,gzip,hashlib,json
from pathlib import Path
import numpy as np

def main():
    report={'runs':{},'issues':[]}
    for name in ['final','kl_baseline','sensitivity','informed','recent_baselines','recent_privic_thinning']:
        run=Path('results')/name;cfg=json.loads((run/'config.json').read_text());rows=list(csv.DictReader((run/'rows.csv').open()));events=0;cases=0;issues=[]
        index={(r['condition'],int(r['seed']),int(r['user'])):r for r in rows}
        for c in cfg['conditions']:
            with gzip.open(run/(c['id']+'.jsonl.gz'),'rt') as f:
                for line in f:
                    case=json.loads(line);ev=case['events'];cases+=1;events+=len(ev)
                    r=index[(c['id'],case['seed'],int(case['user']))]
                    if len(ev)!=cfg['slots']:issues.append('incorrect event count')
                    for metric in ['hit','error_cells','peak','logloss','covered95','credible95','entropy']:
                        if not np.isclose(np.mean([e[metric] for e in ev]),float(r[metric]),rtol=1e-10,atol=1e-12):issues.append(c['id']+':'+metric)
                    for e in ev:
                        if not np.isclose(sum(e['belief']),1,atol=1e-10):issues.append('belief not normalized')
                        q=e['q'][e['evaluator_truth']] if isinstance(e['q'],list) else e['q']
                        if e['reported'] and not(e['eligible'] and e['available'] and q>0):issues.append('invalid truthful completion')
                    complete=sum(e['reported'] and e['legitimate'] for e in ev)
                    opp=sum(e['eligible'] and e['available'] and e['legitimate'] for e in ev)
                    if complete!=int(r['complete']) or opp!=int(r['opportunities']):issues.append('utility counts differ')
        if cases!=len(rows):issues.append('row/log case mismatch')
        if json.loads((run/'failures.json').read_text()):issues.append('recorded failed cases')
        report['runs'][name]={'conditions':len(cfg['conditions']),'trajectories':cases,'events':events,'rows_sha256':hashlib.sha256((run/'rows.csv').read_bytes()).hexdigest(),'issues':issues}
        report['issues']+=issues
    manifest=json.loads(Path('data/manifest.json').read_text())
    report['dataset_sha256_matches']=hashlib.sha256(Path('data/geolife.zip').read_bytes()).hexdigest()==manifest['source_sha256']
    if not report['dataset_sha256_matches']:report['issues'].append('dataset hash mismatch')
    report['scope']='All completed main, KL, sensitivity, informed and recent-baseline logs checked against trajectory rows; this is not independent scientific validation.'
    Path('results/audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert not report['issues']
if __name__=='__main__':main()
