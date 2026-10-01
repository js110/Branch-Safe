"""Analyze data-driven R-BSP and disclosure-floor-aware BSP extensions."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from .geolife import effective_geolife_splits,effective_state_group_by_user

def load_rows(run):
    rows=list(csv.DictReader((run/'rows.csv').open()))
    text={'condition','scenario','method','attack','user'}
    for r in rows:
        for k,v in list(r.items()):
            if k in text:continue
            if v in ('','None'):r[k]=float('nan')
            else:
                try:r[k]=float(v)
                except ValueError:pass
    return rows

def safe_ratio(a,b):
    return float(a/b) if b else float('nan')

def aggregate(rows,slots):
    n=len(rows)
    return dict(
        trajectories=n,
        hit=float(np.mean([r['hit'] for r in rows])),
        peak=float(np.mean([r['peak'] for r in rows])),
        logloss=float(np.mean([r['logloss'] for r in rows])),
        utility=safe_ratio(sum(r['complete'] for r in rows),sum(r['opportunities'] for r in rows)),
        weighted_utility=safe_ratio(sum(r['weighted_complete'] for r in rows),sum(r['weighted_opportunities'] for r in rows)),
        small_retention=safe_ratio(sum(r['small_complete'] for r in rows),sum(r['small_opportunities'] for r in rows)),
        medium_retention=safe_ratio(sum(r['medium_complete'] for r in rows),sum(r['medium_opportunities'] for r in rows)),
        large_retention=safe_ratio(sum(r['large_complete'] for r in rows),sum(r['large_opportunities'] for r in rows)),
        small_opportunities=int(sum(r['small_opportunities'] for r in rows)),
        local_violation_rate=float(sum(r['attacker_local_cap_violations'] for r in rows)/(n*slots)),
        effective_violation_rate=float(sum(r['attacker_effective_cap_violations'] for r in rows)/(n*slots)),
        absolute_violation_rate=float(sum(r['attacker_absolute_cap_violations'] for r in rows)/(n*slots)),
        gate_us_p50=float(np.mean([r['gate_us_p50'] for r in rows])),
    )

def bootstrap(group,slots,scenario,reps=1000,seed=270927):
    geo_map=effective_state_group_by_user("test") if scenario=="geolife" else None
    def cluster_key(r):
        return geo_map[int(r["user"])] if geo_map is not None else int(r["seed"])
    keys=sorted(set(cluster_key(r) for r in group),key=str)
    by={k:[r for r in group if cluster_key(r)==k] for k in keys}
    rng=np.random.default_rng(seed);metrics={}
    for _ in range(reps):
        sample=[]
        for k in rng.choice(keys,size=len(keys),replace=True):sample.extend(by[k])
        a=aggregate(sample,slots)
        for name,val in a.items():
            if name in ('trajectories','small_opportunities'):continue
            metrics.setdefault(name,[]).append(val)
    return {k:(float(np.nanquantile(v,.025)),float(np.nanquantile(v,.975))) for k,v in metrics.items()}

def pct(x):
    return '--' if not np.isfinite(x) else f'{100*x:.1f}'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='results/data_driven_extension')
    ap.add_argument('--paper-assets',action='store_true')
    a=ap.parse_args();run=Path(a.run)
    cfg=json.loads((run/'config.json').read_text());rows=load_rows(run);slots=int(cfg['slots'])
    cond={c['id']:c for c in cfg['conditions']};summary=[]
    for cid in dict.fromkeys(r['condition'] for r in rows):
        g=[r for r in rows if r['condition']==cid];base=aggregate(g,slots);ci=bootstrap(g,slots,g[0]['scenario'])
        d=dict(condition=cid,scenario=g[0]['scenario'],method=g[0]['method'],param=float(g[0]['param']),
               positive_cap=cond[cid].get('positive_cap'),robust_model_count=int(g[0]['robust_model_count']))
        d.update(base)
        for k,(lo,hi) in ci.items():d[k+'_low']=lo;d[k+'_high']=hi
        summary.append(d)
    out=run/'analysis';out.mkdir(exist_ok=True)
    fields=list(summary[0])
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    fit=json.loads(Path(cfg['ambiguity_models_file']).read_text())
    report=dict(study_status=cfg['study_status'],ambiguity_fit=fit,conditions=summary,
                weighted_utility='sum(completed/area)/sum(opportunity/area)',
                area_bins={'small':'1-8','medium':'9-31','large':'32-64'},
                df_interpretation='local_violation_rate tests the original BSP local cap max(rho, prior peak); effective_violation_rate uses max(tau, prior peak) for reports and max(rho, prior peak) for silence')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    geo=[s for s in summary if s['scenario']=='geolife']
    selected=[next(s for s in geo if s['condition']==cid) for cid in
              ['geolife_dd_nominal_bsp','geolife_data_single_bsp','geolife_hand_rbsp','geolife_data_rbsp']]
    names={'geolife_dd_nominal_bsp':'Uniform-prior BSP','geolife_data_single_bsp':'Validation-best single BSP',
           'geolife_hand_rbsp':'Hand-grid R-BSP','geolife_data_rbsp':'Data-driven R-BSP'}
    lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',
           r'Policy & MAP hit & Local viol. & Ret. & Weighted ret. & Small ret. \\',r'\midrule']
    for s in selected:
        lines.append(f"{names[s['condition']]} & {pct(s['hit'])}\% & {pct(s['local_violation_rate'])}\% & "
                     f"{pct(s['utility'])}\% & {pct(s['weighted_utility'])}\% & {pct(s['small_retention'])}\% " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (out/'data_driven_table.tex').write_text('\n'.join(lines)+'\n')

    lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Scenario & Policy & $\tau$ & BSP local viol. & Branch viol. & Weighted ret. & Small ret. \\',r'\midrule']
    for scenario in ['walk','geolife']:
        subset=[s for s in summary if s['scenario']==scenario and s['method'] in ('bsp','dfbsp','rdfbsp')]
        def order(s):
            base=0 if s['method']=='bsp' else (1 if s['method']=='dfbsp' else 2)
            return (float('inf') if s['positive_cap'] is None else float(s['positive_cap']),base)
        for s in sorted(subset,key=order):
            if s['method']=='bsp':
                policy={'geolife_dd_nominal_bsp':'Uniform BSP',
                        'geolife_data_single_bsp':'Val.-best BSP',
                        'walk_new_bsp':'BSP'}.get(s['condition'],'BSP')
            else:
                policy={'dfbsp':'DF-BSP','rdfbsp':'RDF-BSP'}[s['method']]
            tau='--' if s['positive_cap'] is None else f"{float(s['positive_cap']):.3g}"
            lines.append(f"{('GeoLife' if scenario=='geolife' else scenario.capitalize())} & {policy} & {tau} & {pct(s['local_violation_rate'])}\% & "
                         f"{pct(s['effective_violation_rate'])}\% & {pct(s['weighted_utility'])}\% & {pct(s['small_retention'])}\% " + r'\\')
        if scenario!='geolife':lines.append(r'\addlinespace')
    lines += [r'\bottomrule',r'\end{tabular}']
    (out/'dfbsp_table.tex').write_text('\n'.join(lines)+'\n')
    # Generate manuscript macros from the same fitted model and replay summary
    # used for the tables, so prose cannot drift from regenerated evidence.
    splits,_=effective_geolife_splits()
    def row(cid):
        return next(x for x in summary if x['condition']==cid)
    strict=row('geolife_data_rbsp')
    rdf=row('geolife_data_rdfbsp_tau_0.5')
    df=row('geolife_dfbsp_tau_0.5')
    nominal=row('geolife_dd_nominal_bsp')
    single=row('geolife_data_single_bsp')
    hand=row('geolife_hand_rbsp')
    dev=fit['development'];val=fit['validation']
    macros={
        'GeoDevN':len(splits['development']['users']),
        'GeoValN':len(splits['validation']['users']),
        'GeoTestN':len(splits['test']['users']),
        'GeoDevMLE':f"{dev['estimate']:.4f}",
        'GeoDevCILow':f"{dev['ci'][0]:.4f}",
        'GeoDevCIHigh':f"{dev['ci'][1]:.4f}",
        'GeoValMLE':f"{val['estimate']:.4f}",
        'GeoNominalLocal':pct(nominal['local_violation_rate']),
        'GeoSingleLocal':pct(single['local_violation_rate']),
        'GeoSingleRet':pct(single['utility']),
        'GeoHandLocal':pct(hand['local_violation_rate']),
        'GeoStrictHit':pct(strict['hit']),
        'GeoStrictRet':pct(strict['utility']),
        'GeoStrictWeighted':pct(strict['weighted_utility']),
        'GeoStrictLocal':pct(strict['local_violation_rate']),
        'GeoRDFSmall':pct(rdf['small_retention']),
        'GeoRDFSmallLow':pct(rdf['small_retention_low']),
        'GeoRDFSmallHigh':pct(rdf['small_retention_high']),
        'GeoRDFWeighted':pct(rdf['weighted_utility']),
        'GeoRDFBranch':f"{100*rdf['effective_violation_rate']:.2f}",
        'GeoRDFBranchLow':f"{100*rdf['effective_violation_rate_low']:.2f}",
        'GeoRDFBranchHigh':f"{100*rdf['effective_violation_rate_high']:.2f}",
        'GeoRDFLocal':pct(rdf['local_violation_rate']),
        'GeoDFSmall':pct(df['small_retention']),
        'GeoDFWeighted':pct(df['weighted_utility']),
        'GeoDFBranch':pct(df['effective_violation_rate']),
    }
    macro_text='\n'.join('\\newcommand{\\%s}{%s}'%(k,v) for k,v in macros.items())+'\n'
    (out/'current_numbers.tex').write_text(macro_text)
    if a.paper_assets:
        for name in ['data_driven_table.tex','dfbsp_table.tex','current_numbers.tex']:
            shutil.copyfile(out/name,Path('paper')/name)
    print(json.dumps({'ambiguity':{'candidate_moves':fit['move_candidates'],
                                   'selected_models':len(fit['models']),
                                   'selection_mass':fit.get('selection_mass_achieved')},
                      'conditions':len(summary)},indent=2))

if __name__=='__main__':main()
