"""Supplementary sensitivity analysis; no selection of best thresholds."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from .analyze import load,clusters,COLORS,MARKERS,LABELS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summary(rows):
    out=[]
    for cid in dict.fromkeys(r['condition'] for r in rows):
        a=[r for r in rows if r['condition']==cid];keys,cs=clusters(a)
        rng=np.random.default_rng(92026);idx=rng.integers(len(cs),size=(1000,len(cs)))
        d={k:a[0][k] for k in ['condition','method','param','side']};d['n_clusters']=len(cs)
        for metric in ['hit','peak','covered95','error_cells','logloss','ece','gate_us_p50','gate_us_p95']:
            v=np.array([c[metric] for c in cs]);draws=v[idx].mean(axis=1)
            d[metric]=float(v.mean());d[metric+'_low'],d[metric+'_high']=np.quantile(draws,[.025,.975])
        v=np.array([c['complete'] for c in cs]);den=np.array([c['opportunities'] for c in cs]);draws=v[idx].sum(axis=1)/den[idx].sum(axis=1)
        d['utility']=v.sum()/den.sum();d['utility_low'],d['utility_high']=np.quantile(draws,[.025,.975]);out.append(d)
    return out

def main():
    summaries={}
    for name in ['sensitivity','informed']:
        run=Path('results')/name;out=run/'analysis';out.mkdir(exist_ok=True);summaries[name]=summary(load(run/'rows.csv'))
        with (out/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=summaries[name][0]);w.writeheader();w.writerows(summaries[name])
        (out/'provenance.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256((run/'rows.csv').read_bytes()).hexdigest(),bootstrap=1000,clusters='10 synthetic seeds, four trajectories per seed',interval='pointwise percentile 95%'),indent=2))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    for ax,prefix,label in zip(axs.ravel(),['move_true','alpha_model','reset_every','side'],['True move probability','Assumed availability probability','Ideal identity reset period (slots)','Grid side length']):
        for method in ['none','random','bsp']:
            a=[s for s in summaries['sensitivity'] if s['condition'].startswith(prefix+'_') and s['method']==method]
            a.sort(key=lambda s:float(s['condition'][len(prefix)+1:].rsplit('_',1)[0]));x=[float(s['condition'][len(prefix)+1:].rsplit('_',1)[0]) for s in a]
            ax.errorbar(x,[s['hit'] for s in a],yerr=[[s['hit']-s['hit_low'] for s in a],[s['hit_high']-s['hit'] for s in a]],label=LABELS[method],color=COLORS[method],marker=MARKERS[method],capsize=2,lw=1)
        ax.set(xlabel=label,ylabel='MAP cell hit rate');ax.grid(alpha=.15)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
    out=Path('results/sensitivity/analysis');fig.savefig(out/'sensitivity.pdf');fig.savefig(out/'sensitivity.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
    for ax,metric,ylab in zip(axs,['hit','peak'],['MAP cell hit rate','Mean attacker posterior maximum']):
        for run,ls,marker in [('sensitivity','--','o'),('informed','-','s')]:
            a=[s for s in summaries[run] if s['method']=='bsp' and ('alpha_model' in s['condition'] or 'move_model' in s['condition'])]
            ax.plot(range(4),[s[metric] for s in a],ls=ls,marker=marker,label='Shared client model' if run=='sensitivity' else 'Better-informed attacker',color='#D55E00' if run=='informed' else '#0072B2')
        ax.set_xticks(range(4),['a=.3','a=.9','v=.05','v=.8']);ax.set(xlabel='Misspecified client parameter',ylabel=ylab);ax.grid(alpha=.15)
        if metric=='peak':ax.axhline(.1,c='gray',ls=':',label='Client threshold')
    h,l=axs[1].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    out=Path('results/informed/analysis');fig.savefig(out/'informed.pdf');fig.savefig(out/'informed.png');plt.close(fig)
    print('Analyzed stress conditions', {k:len(v) for k,v in summaries.items()})
if __name__=='__main__':main()
