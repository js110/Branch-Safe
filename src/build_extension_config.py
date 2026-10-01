"""Build the frozen extension experiment from a fitted ambiguity-set file."""
import argparse,json
from pathlib import Path


def base_condition(method,param,scenario,move_model,tag,positive_cap=None):
    c=dict(delivery=.9,willing=.8,on_time=.9,move_true=.3,move_model=float(move_model),
           side=8,scenario=scenario,attack='adaptive',method=method,param=float(param),
           alpha_model=.648,id=tag)
    if positive_cap is not None:c['positive_cap']=float(positive_cap)
    return c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',default='results/data_driven_ambiguity/models.json')
    ap.add_argument('--output',default='configs/data_driven_extension.json')
    a=ap.parse_args()
    fit=json.loads(Path(a.models).read_text())
    dev=float(fit['development']['estimate']);val=float(fit['validation']['estimate'])
    pooled=float(fit['pooled']['estimate']);pop=fit['population_prior']
    selected=fit.get('candidate_models',[])
    winners=[m for m in selected if m.get('selected')]
    best=max(winners,key=lambda m:m.get('validation_selection_frequency',0)) if winners else dict(move=pooled)
    best_move=float(best['move'])
    conditions=[]
    # GeoLife: validation-informed attacker; test users are not used in fitting.
    def geo(method,tag,tau=None,robust_file=None,models=None):
        c=base_condition(method,.1,'geolife',dev,tag,tau)
        c.update(attacker_move=val,attacker_alpha=.648,attacker_prior=pop)
        if robust_file:c['robust_models_file']=robust_file
        if models is not None:c['robust_models']=models
        return c
    conditions.append(geo('bsp','geolife_dd_nominal_bsp'))
    single=geo('bsp','geolife_data_single_bsp')
    single['move_model']=best_move;single['client_prior']=pop
    conditions.append(single)
    hand=[dict(move=m,alpha=aa) for m in [.05,.3,.8] for aa in [.3,.648,.9]]
    conditions.append(geo('rbsp','geolife_hand_rbsp',models=hand))
    conditions.append(geo('rbsp','geolife_data_rbsp',robust_file=a.models))
    for tau in [.125,.25,.5,1.0]:
        conditions.append(geo('dfbsp',f'geolife_dfbsp_tau_{tau}',tau=tau))
        conditions.append(geo('rdfbsp',f'geolife_data_rdfbsp_tau_{tau}',tau=tau,robust_file=a.models))
    # Independent synthetic seeds: isolate the service/privacy effect without prior mismatch.
    conditions.append(base_condition('bsp',.1,'walk',.3,'walk_new_bsp'))
    for tau in [.125,.25,.5,1.0]:
        conditions.append(base_condition('dfbsp',.1,'walk',.3,f'walk_new_dfbsp_tau_{tau}',tau))
    cfg=dict(
        output='results/data_driven_extension',side=8,slots=48,
        seeds=list(range(5000,5020)),users_per_seed=4,
        geolife_file='data/geolife_test.npz',
        study_status='post-hoc extension; ambiguity set fitted only on development/validation; GeoLife test previously inspected in earlier analyses',
        ambiguity_models_file=a.models,
        attacker_model=dict(move=val,alpha=.648,prior='population_smoothed_from_development'),
        nominal_move=dev,validation_best_single_move=best_move,pooled_move=pooled,
        disclosure_floor_positive_caps=[.125,.25,.5,1.0],
        conditions=conditions)
    Path(a.output).write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(conditions=len(conditions),development_move=dev,validation_attacker_move=val,pooled_move=pooled),indent=2))


if __name__=='__main__':main()
