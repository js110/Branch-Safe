"""Recent-baseline curves and paired cluster intervals, without changing old results."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze import load,clusters,METRICS,aggregate

def lower_hull(points):
    """Lower convex envelope of measured (utility, risk) policy points."""
    by_x={}
    for x,y in points:
        if np.isfinite(x) and np.isfinite(y):by_x[x]=min(y,by_x.get(x,float('inf')))
    hull=[]
    for p in sorted(by_x.items()):
        while len(hull)>1:
            a,b=hull[-2:]
            cross=(b[0]-a[0])*(p[1]-b[1])-(b[1]-a[1])*(p[0]-b[0])
            if cross>0:break
            hull.pop()
        hull.append(p)
    return hull

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recent-run',default='results/recent_baselines')
    parser.add_argument('--reference-run',default='results/final');parser.add_argument('--output',default='results/recent_comparison')
    parser.add_argument('--paper-assets',action='store_true')
    parser.add_argument('--extra-run',nargs='*',default=['results/recent_privic_thinning'])
    a=parser.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    paths=[Path(a.recent_run)/'rows.csv',Path(a.reference_run)/'rows.csv']+[Path(p)/'rows.csv' for p in a.extra_run];rows=load(paths[0])
    for path in paths[2:]:rows+=load(path)
    rows += [r for r in load(paths[1]) if r['attack']=='adaptive' and r['method'] in ['bsp','random','none']]
    summaries=[];boots={};keysets={};nboot=1000
    for cid in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==cid];keys,cs=clusters(group);keysets[cid]=keys
        rng=np.random.default_rng(92026);indices=rng.integers(len(cs),size=(nboot,len(cs)))
        samples=[aggregate(cs,idx) for idx in indices]
        central=aggregate(cs)
        s={k:group[0][k] for k in ['condition','scenario','method','param','attack','side']}
        s.update(n_clusters=len(cs),n_trajectories=len(group))
        for k in METRICS:
            vals=np.asarray([x[k] for x in samples],dtype=float)
            s[k]=central[k];s[k+'_low'],s[k+'_high']=np.nanquantile(vals,[.025,.975])
        u=np.asarray([x['utility'] for x in samples],dtype=float)
        s['utility']=central['utility'];s['utility_low'],s['utility_high']=np.nanquantile(u,[.025,.975])
        summaries.append(s);boots[cid]=(u,np.asarray([x['hit'] for x in samples],dtype=float))
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    matched=[];raw_matched=[]
    for sc in ['static','walk','commute','geolife']:
        curves={m:[s for s in summaries if s['scenario']==sc and s['method']==m] for m in ['bsp','pml','privic','random']}
        for target in [.5,.7]:
            def at(curve,draw=None,envelope=True):
                pts=sorted((s['utility'],s['hit']) if draw is None else (boots[s['condition']][0][draw],boots[s['condition']][1][draw]) for s in curve)
                if envelope:pts=lower_hull(pts)
                if not pts[0][0]<=target<=pts[-1][0]:return float('nan')
                return float(np.interp(target,[x for x,y in pts],[y for x,y in pts]))
            for rival in ['pml','privic','random']:
                allkeys=[keysets[s['condition']] for s in curves['bsp']+curves[rival]]
                assert all(k==allkeys[0] for k in allkeys),'unpaired cluster keys'
                bv=at(curves['bsp']);rv=at(curves[rival]);diff=bv-rv
                ds=np.array([at(curves['bsp'],i)-at(curves[rival],i) for i in range(nboot)]);ds=ds[np.isfinite(ds)]
                lo,hi=np.quantile(ds,[.025,.975]) if len(ds)>=.95*nboot and np.isfinite(diff) else (np.nan,np.nan)
                matched.append(dict(scenario=sc,target_utility=target,comparator=rival,bsp_hit=bv,comparator_hit=rv,
                                    difference=diff,low=lo,high=hi,valid_bootstraps=len(ds)))
                raw_matched.append(dict(scenario=sc,target_utility=target,comparator=rival,bsp_hit=at(curves['bsp'],envelope=False),comparator_hit=at(curves[rival],envelope=False)))
    clean=[{k:float(v) if isinstance(v,(np.floating,float)) and np.isfinite(v) else None if isinstance(v,(np.floating,float)) else v for k,v in r.items()} for r in matched]
    (out/'matched_utility.json').write_text(json.dumps(clean,indent=2,allow_nan=False))
    (out/'adjacent_interpolation.json').write_text(json.dumps([{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()} for r in raw_matched],indent=2,allow_nan=False))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
    styles={'bsp':('BSP','#D55E00','^'),'pml':('PML-T (TIFS 2024)','#009E73','s'),
            'privic':('PRIVIC-T (PoPETs 2024)','#0072B2','D'),'random':('Random participation','#999999','o')}
    for ax,sc in zip(axs.ravel(),['static','walk','commute','geolife']):
        for m,(label,color,marker) in styles.items():
            ss=sorted([s for s in summaries if s['scenario']==sc and s['method']==m],key=lambda s:s['utility'])
            pts=lower_hull([(s['utility'],s['hit']) for s in ss])
            ax.scatter([s['utility'] for s in ss],[s['hit'] for s in ss],marker=marker,color=color,s=16,alpha=.25)
            ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=marker,color=color,label=label,lw=1.2,ms=4)
            selected=[next(s for s in ss if s['utility']==x and s['hit']==y) for x,y in pts]
            ax.errorbar([s['utility'] for s in selected],[s['hit'] for s in selected],yerr=[[max(0,s['hit']-s['hit_low']) for s in selected],[max(0,s['hit_high']-s['hit']) for s in selected]],fmt='none',ecolor=color,alpha=.45,lw=.65)
        ax.axhline(1/64,c='#777777',ls=':',lw=.8)
        ax.set(title=sc.capitalize(),xlabel='Legitimate opportunity retention',ylabel='MAP cell hit rate',xlim=(0,1.03),ylim=(0,.65));ax.grid(alpha=.15)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(out/'recent_tradeoff.pdf');fig.savefig(out/'recent_tradeoff.png');plt.close(fig)
    lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Scenario & Comparator & BSP hit & Other hit & Difference [95\% CI] \\',r'\midrule']
    for r in clean:
        if r['target_utility']!=.5 or r['comparator']=='random':continue
        v=lambda k:'--' if r[k] is None else f'{100*r[k]:.2f}'
        interval='not estimable' if r['low'] is None else f"{100*r['difference']:+.2f} [{100*r['low']:+.2f}, {100*r['high']:+.2f}]"
        lines.append(f"{r['scenario'].capitalize()} & "+('PML-T' if r['comparator']=='pml' else 'PRIVIC-T')+f" & {v('bsp_hit')} & {v('comparator_hit')} & {interval} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(out/'recent_matched_table.tex').write_text('\n'.join(lines))
    provenance=dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},bootstrap=nboot,seed=92026,
                    unit='point estimates average trajectory/window rows; uncertainty resamples synthetic seeds or GeoLife complete 48-slot state-path groups',ci='paired cluster-percentile, pointwise, no multiplicity correction',
                    interpolation='lower convex envelope, reselected within each bootstrap; inside observed range only; require 95% overlap; exploratory oracle mixtures, no deployed-policy claim',conditions=len(summaries),trajectory_rows=len(rows))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if a.paper_assets:
        shutil.copyfile(out/'recent_tradeoff.pdf',Path('paper')/'recent_tradeoff.pdf')
    print(json.dumps(clean,indent=2))

if __name__=='__main__':main()
