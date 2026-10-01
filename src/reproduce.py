"""Run existing frozen configurations into a fresh directory without overwriting evidence."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
from .experiment import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--runs',nargs='+',default=['final','kl_baseline','sensitivity','informed','recent_baselines','recent_privic_thinning']);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    for name in a.runs:
        cfg=json.loads(Path('configs',name+'.json').read_text());cfg['output']=str(root/name)
        if name in ['recent_baselines','recent_privic_thinning']:
            from .recent_baselines import prepare
            training=root/'privic_training'
            if not (training/'manifest.json').exists():prepare('configs/recent_baselines.json',training)
            cfg['training_output']=str(training)
            for cond in cfg['conditions']:
                if cond['method']=='privic':cond['channel_file']=str(training/Path(cond['channel_file']).name)
        path=root/'configs'/(name+'.json');path.write_text(json.dumps(cfg,indent=2));run(path)
    if all(name in a.runs for name in ['final','kl_baseline']):
        out=root/'combined';out.mkdir();rows=[];inputs={}
        for name in ['final','kl_baseline']:
            path=root/name/'rows.csv';rows.extend(csv.DictReader(path.open()));inputs[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with (out/'rows.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        (out/'inputs.json').write_text(json.dumps(inputs,indent=2));subprocess.run([sys.executable,'-m','src.analyze','--run',str(out)],check=True)
    if all(name in a.runs for name in ['final','recent_baselines']):
        extras=[str(root/'recent_privic_thinning')] if 'recent_privic_thinning' in a.runs else []
        subprocess.run([sys.executable,'-m','src.analyze_recent','--reference-run',str(root/'final'),
                        '--recent-run',str(root/'recent_baselines'),'--output',str(root/'recent_comparison'),'--extra-run',*extras],check=True)
if __name__=='__main__':main()
