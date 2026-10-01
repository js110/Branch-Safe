"""Cluster bootstrap analysis; all curves and tables originate from rows.csv."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .geolife import effective_state_group_by_user

METRICS=['hit','error_cells','error_m','peak','logloss','covered95','credible95','entropy','prior_hit','raw_completion','coverage','reward','gate_us_p50','gate_us_p95','ece']
COLORS={'none':'#222222','random':'#0072B2','rate':'#E69F00','coarse':'#009E73','bsp':'#D55E00','positive_only':'#CC79A7','kl':'#56B4E9'}
MARKERS={'none':'*','random':'o','rate':'s','coarse':'D','bsp':'^','positive_only':'x','kl':'P'}
LABELS={'none':'No filter','random':'Random participation','rate':'Periodic limit','coarse':'Aligned task grid','bsp':'BSP','positive_only':'Success-only ablation','kl':'KL branch filter'}

def load(path):
    rows=list(csv.DictReader(Path(path).open()))
    for r in rows:
        for k in list(r):
            if k not in ['condition','scenario','method','attack','user']:
                v=r[k]
                r[k]=float('nan') if v in ('','None') else float(v)
    return rows

def clusters(rows):
    geo_map=effective_state_group_by_user("test") if rows and rows[0]["scenario"]=="geolife" else None
    def cluster_key(r):
        return geo_map[int(r["user"])] if geo_map is not None else int(r["seed"])
    keys=sorted(set(cluster_key(r) for r in rows),key=str)
    out=[]
    for key in keys:
        a=[r for r in rows if cluster_key(r)==key]
        d={}
        for k in METRICS:
            vals=np.asarray([r[k] for r in a],dtype=float)
            finite=np.isfinite(vals)
            d[k]=float(np.mean(vals[finite])) if finite.any() else float('nan')
            d['__n_'+k]=int(finite.sum())
        d['complete']=sum(r['complete'] for r in a);d['opportunities']=sum(r['opportunities'] for r in a)
        d['__n_rows']=len(a)
        out.append(d)
    return keys,out

def aggregate(cs,indices=None):
    if indices is None:indices=np.arange(len(cs))
    a=[cs[i] for i in indices]
    out={}
    for k in METRICS:
        weighted=[(r[k],r.get('__n_'+k,r.get('__n_rows',1))) for r in a if np.isfinite(r[k])]
        denom=sum(n for _,n in weighted)
        out[k]=sum(v*n for v,n in weighted)/denom if denom else float('nan')
    denom=sum(r['opportunities'] for r in a)
    out['utility']=sum(r['complete'] for r in a)/denom if denom else float('nan')
    return out

def summarize(rows,nboot=1000):
    summaries=[];cache={}
    for cid in dict.fromkeys(r['condition'] for r in rows):
        a=[r for r in rows if r['condition']==cid];keys,cs=clusters(a);cache[cid]=(keys,cs)
        rng=np.random.default_rng(92026)
        # Same resamples across matched conditions with same cluster IDs.
        samples=[aggregate(cs,rng.integers(len(cs),size=len(cs))) for _ in range(nboot)]
        central=aggregate(cs)
        d={k:a[0][k] for k in ['condition','scenario','method','param','attack','side']}
        d.update(n_clusters=len(cs),n_trajectories=len(a),cap_violations=sum(r['cap_violations'] for r in a))
        for k,v in central.items():
            vals=np.array([s[k] for s in samples]);vals=vals[np.isfinite(vals)]
            d[k]=v;d[k+'_low']=float(np.quantile(vals,.025)) if len(vals) else float('nan');d[k+'_high']=float(np.quantile(vals,.975)) if len(vals) else float('nan')
        summaries.append(d)
    return summaries,cache

def matched(rows,cache,summaries,nboot=1000):
    results=[]
    for scenario in ['static','walk','commute','geolife']:
        curves={m:[s for s in summaries if s['scenario']==scenario and s['attack']=='adaptive' and s['method']==m] for m in ['bsp','random','rate','coarse','kl']}
        for target in [.5,.7]:
            for rival in ['random','rate','coarse','kl']:
                if not curves['bsp'] or not curves[rival]:continue
                def value(curve,idx=None):
                    pts=[]
                    for s in curve:
                        cs=cache[s['condition']][1];v=aggregate(cs,idx)
                        pts.append((v['utility'],v['hit']))
                    pts=sorted(pts)
                    if not pts[0][0]<=target<=pts[-1][0]:return float('nan')
                    return float(np.interp(target,[p[0] for p in pts],[p[1] for p in pts]))
                bv=value(curves['bsp']);rv=value(curves[rival])
                n=len(cache[curves['bsp'][0]['condition']][1]);rng=np.random.default_rng(92027)
                draws=[]
                for _ in range(nboot):
                    idx=rng.integers(n,size=n);d=value(curves['bsp'],idx)-value(curves[rival],idx)
                    if np.isfinite(d):draws.append(d)
                results.append(dict(scenario=scenario,target_utility=target,comparator=rival,bsp_hit=bv,comparator_hit=rv,difference=bv-rv,
                    low=float(np.quantile(draws,.025)) if len(draws)>=.95*nboot and np.isfinite(bv-rv) else None,high=float(np.quantile(draws,.975)) if len(draws)>=.95*nboot and np.isfinite(bv-rv) else None,valid_bootstraps=len(draws),
                    interpretation='linear interpolation inside observed utility range only; exploratory, not a deployed matched policy'))
    return results

def figures(summaries,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.0),layout='constrained')
    for ax,scenario in zip(axs.ravel(),['static','walk','commute','geolife']):
        for method in LABELS:
            a=sorted([s for s in summaries if s['scenario']==scenario and s['attack']=='adaptive' and s['method']==method],key=lambda s:s['utility'])
            if not a:continue
            ax.plot([s['utility'] for s in a],[s['hit'] for s in a],color=COLORS[method],marker=MARKERS[method],label=LABELS[method],lw=1.2,ms=5)
            ax.errorbar([s['utility'] for s in a],[s['hit'] for s in a],yerr=[[max(0,s['hit']-s['hit_low']) for s in a],[max(0,s['hit_high']-s['hit']) for s in a]],fmt='none',ecolor=COLORS[method],alpha=.5,lw=.7)
        ax.set(title=scenario.capitalize(),xlabel='Legitimate opportunity retention',ylabel='MAP cell hit rate',xlim=(0,1.04),ylim=(0,.65));ax.grid(alpha=.15)
        ax.axhline(1/64,color='gray',ls=':',lw=.8)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(out/'tradeoff.pdf');fig.savefig(out/'tradeoff.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for method in ['none','random','bsp','positive_only']:
        par={'none':1,'random':.5,'bsp':.1,'positive_only':.1}[method]
        a=[next(s for s in summaries if s['scenario']==sc and s['attack']=='adaptive' and s['method']==method and s['param']==par) for sc in ['static','walk','commute','geolife']]
        for ax,k in zip(axs,['covered95','peak']):ax.plot(range(4),[s[k] for s in a],marker=MARKERS[method],color=COLORS[method],label=LABELS[method])
    axs[0].axhline(.95,c='gray',ls=':');axs[0].set(ylabel='Empirical 95% set coverage',ylim=(0,1.02))
    axs[1].set(ylabel='Mean posterior maximum',ylim=(0,.7))
    for ax in axs:ax.set_xticks(range(4),['Static','Walk','Commute','GeoLife']);ax.grid(alpha=.15)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(out/'calibration.pdf');fig.savefig(out/'calibration.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3),layout='constrained')
    x=np.arange(4);scs=['static','walk','commute','geolife']
    for j,(method,attack) in enumerate([('none','fixed'),('none','adaptive'),('bsp','fixed'),('bsp','adaptive')]):
        a=[next(s for s in summaries if s['scenario']==sc and s['attack']==attack and s['method']==method and s['param']==(1 if method=='none' else .1)) for sc in scs]
        ax.bar(x+(j-1.5)*.19,[s['hit'] for s in a],.18,label=f'{LABELS[method]}, {attack}',color=['#777777','#222222','#E69F00','#D55E00'][j],hatch='//' if attack=='fixed' else None)
    ax.set_xticks(x,['Static','Walk','Commute','GeoLife']);ax.set(ylabel='MAP cell hit rate',ylim=(0,.7));ax.legend(fontsize=8,ncol=2);fig.savefig(out/'attacks.pdf');fig.savefig(out/'attacks.png');plt.close(fig)

def write_tables(summaries,out):
    lines=['\\begin{tabular}{llrrrr}','\\toprule','Scenario & Policy & Retention & MAP hit & Error (cells) & 95\\% cover \\\\','\\midrule']
    for sc in ['static','walk','commute','geolife']:
        for method,par in [('none',1),('random',.5),('rate',2),('coarse',2),('positive_only',.1),('bsp',.1)]:
            s=next(s for s in summaries if s['scenario']==sc and s['attack']=='adaptive' and s['method']==method and s['param']==par)
            name={'none':'None','random':'Random .5','rate':'Period 2','coarse':'Grid 2','positive_only':'Success only','bsp':'BSP .1'}[method]
            lines.append(f"{sc.capitalize()} & {name} & {s['utility']:.3f} & {s['hit']:.3f} & {s['error_cells']:.2f} & {s['covered95']:.3f} \\\\")
        if sc!='geolife':lines.append('\\addlinespace')
    lines.extend(['\\bottomrule','\\end{tabular}'])
    (out/'main_table.tex').write_text('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='results/final');args=ap.parse_args();run=Path(args.run)
    rows=load(run/'rows.csv');summaries,cache=summarize(rows);out=run/'analysis';out.mkdir(exist_ok=True)
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    (out/'matched_utility.json').write_text(json.dumps([{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()} for r in matched(rows,cache,summaries)],indent=2,allow_nan=False))
    figures(summaries,out);write_tables(summaries,out)
    (out/'provenance.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256((run/'rows.csv').read_bytes()).hexdigest(),bootstrap=1000,
        unit='point estimates: trajectory/window rows; uncertainty clusters: synthetic seed or GeoLife complete 48-slot state-path group',estimator='window-weighted mean for risk metrics; ratio of summed completions to opportunities for utility; cluster bootstrap for intervals',
        figure_files=['tradeoff','calibration','attacks'],interval='percentile cluster bootstrap 95%, pointwise; no multiplicity correction'),indent=2))
    print('Analyzed',len(rows),'rows and',len(summaries),'conditions')
if __name__=='__main__':main()
