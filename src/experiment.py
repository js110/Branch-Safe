"""Reproducible event simulation. Evaluation truth is never passed to the attack."""
import argparse,csv,gzip,hashlib,json,platform,time
from pathlib import Path
import numpy as np
from functools import lru_cache
from .model import Task,Observation,PublicBelief,RobustPublicBelief,predict,update,normalize,gates,df_bsp_gates,robust_bsp_gates,robust_df_bsp_gates,disclosure_floor,binary_entropy,choose_probe,posterior_metrics
from .geolife import load_effective_geolife

@lru_cache(maxsize=64)
def load_channel(path):
    return np.load(path)['channel']

@lru_cache(maxsize=16)
def load_robust_models(path):
    d=json.loads(Path(path).read_text())
    return d['models'] if isinstance(d,dict) else d

def candidates(side):
    rects=set()
    for width in sorted(set([1,2,max(2,side//2),max(2,3*side//4),side])):
        for x in range(0,side-width+1,max(1,width//2)):
            for y in range(0,side-width+1,max(1,width//2)):
                rects.add((x,x+width,y,y+width))
    for axis in range(2):
        for k in range(1,side):
            rects.add((0,k,0,side) if axis==0 else (0,side,0,k))
            rects.add((k,side,0,side) if axis==0 else (0,side,k,side))
    rects=sorted(rects)
    masks=np.array([Task('',0,0,r).mask(side) for r in rects],float)
    return rects,masks

def _value_at(spec,t):
    a=np.asarray(spec,dtype=float)
    return float(a) if a.ndim==0 else float(a[min(t,len(a)-1)])

def _alpha_vector(alpha,n):
    a=np.asarray(alpha,dtype=float)
    if a.ndim==0:return np.full(n,float(a))
    if a.shape!=(n,):raise ValueError('location-dependent alpha must have one value per state')
    if np.any(a<0) or np.any(a>1):raise ValueError('alpha values must be in [0,1]')
    return a

def path_for(seed,user,side,slots,scenario,move):
    rng=np.random.default_rng(np.random.SeedSequence([seed,user,101]))
    if scenario=='commute':
        y=int(rng.integers(side));offset=int(rng.integers(2*(side-1)))
        v=(np.arange(slots)+offset)%(2*(side-1))
        return (np.minimum(v,2*(side-1)-v)*side+y).astype(int)
    state=int(rng.integers(side*side));out=[]
    for t in range(slots):
        out.append(state)
        move_t=_value_at(move,t)
        if scenario!='static' and rng.random()<move_t:
            x,y=divmod(state,side);axis=int(rng.integers(4))
            dx,dy=[(1,0),(-1,0),(0,1),(0,-1)][axis]
            state=int(np.clip(x+dx,0,side-1)*side+np.clip(y+dy,0,side-1))
    return np.array(out)

def _gate_matrix(condition,client_b,robust_bs,masks,rects,slot,alpha_model,channel,privic_qs):
    method=condition['method'];param=condition['param'];positive_cap=condition.get('positive_cap')
    if channel is not None:return np.asarray(privic_qs,float)
    if robust_bs is not None:
        alphas=np.asarray([m.get('alpha',alpha_model) for m in condition['_robust_specs']],float)
        if method=='rdfbsp':
            q=robust_df_bsp_gates(robust_bs,masks,alphas,param,positive_cap,slot,rects)
        else:q=robust_bsp_gates(robust_bs,masks,alphas,param,slot,rects)
    elif method=='dfbsp':q=df_bsp_gates(client_b,masks,alpha_model,param,positive_cap,slot,rects)
    else:q=gates(client_b,masks,alpha_model,method,param,slot,rects)
    return np.asarray(q,float)[:,None]

def _lookahead2_probe(condition,belief,attacker,robust,masks,rects,slot,alpha_model,move_model,channel,privic_qs):
    """Two-step beam look-ahead over consecutive probe opportunities.

    The score is current mutual information plus the branch-probability-weighted
    best next-slot mutual information. It is a stress heuristic, not an optimal
    long-horizon policy.
    """
    robust_bs=robust.beliefs if robust is not None else None
    qmat=_gate_matrix(condition,belief.b,robust_bs,masks,rects,slot,alpha_model,channel,privic_qs)
    aalpha=_alpha_vector(attacker.alpha,len(attacker.b))
    likelihood=masks*(aalpha[None,:]*qmat)
    mass=(likelihood*attacker.b).sum(axis=1)
    immediate=binary_entropy(mass)-(binary_entropy(likelihood)*attacker.b).sum(axis=1)
    beam=max(1,min(int(condition.get('lookahead_beam',12)),len(masks)))
    beam_idx=np.argsort(-immediate,kind='stable')[:beam]
    discount=float(condition.get('lookahead_discount',1.0))
    attacker_move_next=_value_at(condition.get('attacker_move_schedule',condition.get('attacker_move',move_model)),slot+1)
    scores=[]
    for idx in beam_idx:
        qrow=qmat[idx]
        mask=masks[idx]
        p1=float(mass[idx])
        future=0.0
        for reported,py in ((True,p1),(False,1-p1)):
            if py<=1e-14:continue
            ab=update(attacker.b,mask,aalpha,qrow,reported)
            ab=predict(ab,attacker.side,attacker_move_next)
            cb=update(belief.b,mask,alpha_model,qrow,reported)
            cb=predict(cb,belief.side,move_model)
            rbs=None
            if robust is not None:
                branch=[]
                for member in robust.members:
                    rb=update(member.b,mask,member.alpha,qrow,reported)
                    branch.append(predict(rb,robust.side,member.move))
                rbs=np.vstack(branch)
            nq=_gate_matrix(condition,cb,rbs,masks,rects,slot+1,alpha_model,channel,privic_qs)
            nlike=masks*(aalpha[None,:]*nq)
            nmass=(nlike*ab).sum(axis=1)
            nmi=binary_entropy(nmass)-(binary_entropy(nlike)*ab).sum(axis=1)
            future+=py*float(np.max(nmi))
        scores.append(float(immediate[idx])+discount*future)
    return int(beam_idx[int(np.argmax(scores))])

def simulate(condition,seed,user,path,rects,masks,slots,side,log):
    delivery=condition['delivery'];willing=condition['willing'];on_time=condition['on_time']
    alpha_true=delivery*willing*on_time
    if condition.get('availability_by_cell') is not None:
        alpha_true=float(np.mean(_alpha_vector(condition['availability_by_cell'],side*side)))
    alpha_model=condition.get('alpha_model',alpha_true)
    move_model=condition['move_model']
    method=condition['method'];param=condition['param'];attack=condition['attack']
    positive_cap=condition.get('positive_cap')
    channel=load_channel(condition['channel_file']) if method=='privic' else None
    # This public map contains a gate probability for EVERY possible location.
    # The attacker never receives the probability indexed by the true location.
    gate_scale=condition.get('gate_scale',1.0)
    privic_qs=gate_scale*(masks@channel.T) if channel is not None else None
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(user),202]))
    normal_ids=rng.integers(len(rects),size=slots)
    random_probe_ids=rng.integers(len(rects),size=slots)
    ext=rng.random((slots,4))
    legitimate_schedule=rng.random(slots)<condition.get('legitimate_fraction',.75)
    belief=PublicBelief(side,move_model,alpha_model)
    if condition.get('client_prior') is not None:
        belief.b=np.asarray(condition['client_prior'],float);belief.b=belief.b/belief.b.sum()
    attacker_alpha=condition.get('attacker_alpha_by_cell',condition.get('attacker_alpha',alpha_model))
    attacker=PublicBelief(side,condition.get('attacker_move',move_model),attacker_alpha)
    if condition.get('attacker_prior') is not None:
        attacker.b=np.asarray(condition['attacker_prior'],float);attacker.b=attacker.b/attacker.b.sum()
    robust=None
    if method in ('rbsp','rdfbsp'):
        robust_models=condition.get('robust_models')
        if robust_models is None and condition.get('robust_models_file'):
            robust_models=load_robust_models(condition['robust_models_file'])
        robust_models=robust_models or [dict(move=move_model,alpha=alpha_model)]
        condition['_robust_specs']=robust_models
        robust=RobustPublicBelief(side,robust_models)
    client_background=belief.b.copy()
    attacker_background=attacker.b.copy()
    sums={k:0. for k in ['hit','error_cells','peak','logloss','covered95','credible95','entropy']}
    prior_hit=0.;legit=0;opportunities=0;complete=0;reports=0;cap_violations=0
    attacker_local_cap_violations=0;attacker_effective_cap_violations=0;attacker_absolute_cap_violations=0;op_cells=set();done_cells=set()
    weighted_opportunities=0.;weighted_complete=0.
    area_opp={'small':0,'medium':0,'large':0};area_done={'small':0,'medium':0,'large':0}
    gate_times=[];rows=[];bin_counts=np.zeros(10);bin_hits=np.zeros(10);bin_peaks=np.zeros(10);error_m=0.
    for t in range(slots):
        # Identity resetting is an ideal control: discard past target-specific evidence.
        if condition.get('reset_every',0) and t%condition['reset_every']==0:
            belief.b=client_background.copy()
            attacker.b=attacker_background.copy()
            if robust:robust.reset()
        if t:
            client_move_t=_value_at(condition.get('move_model_schedule',move_model),t)
            attacker_move_t=_value_at(condition.get('attacker_move_schedule',condition.get('attacker_move',move_model)),t)
            belief.b=predict(belief.b,side,client_move_t)
            attacker.b=predict(attacker.b,side,attacker_move_t)
            if robust:robust.predict()
            client_background=predict(client_background,side,client_move_t)
            attacker_background=predict(attacker_background,side,attacker_move_t)
        is_legit=bool(legitimate_schedule[t])
        if is_legit: idx=int(normal_ids[t])
        elif attack=='lookahead2':
            idx=_lookahead2_probe(condition,belief,attacker,robust,masks,rects,t,alpha_model,move_model,channel,privic_qs)
        elif attack=='adaptive':
            if channel is not None:
                qs=privic_qs
            elif robust is not None:
                qs=robust.gates(masks,param,t,rects,positive_cap if method=='rdfbsp' else None)[:,None]
            elif method=='dfbsp':
                qs=df_bsp_gates(belief.b,masks,alpha_model,param,positive_cap,t,rects)[:,None]
            else:
                qs=gates(belief.b,masks,alpha_model,method,param,t,rects)[:,None]
            likelihood=masks*(_alpha_vector(attacker.alpha,side*side)[None,:]*qs)
            information=binary_entropy((likelihood*attacker.b).sum(axis=1))-(binary_entropy(likelihood)*attacker.b).sum(axis=1)
            idx=int(np.argmax(information))
        else:idx=int(random_probe_ids[t])
        task=Task(f'{t}',t,t,rects[idx],is_legit)
        start=time.perf_counter_ns()
        if channel is not None:
            q=gate_scale*(channel@masks[idx])
        elif robust is not None:
            q=float(robust.gates(masks[idx],param,t,[rects[idx]],positive_cap if method=='rdfbsp' else None)[0])
        elif method=='dfbsp':
            q=float(df_bsp_gates(belief.b,masks[idx],alpha_model,param,positive_cap,t,[rects[idx]])[0])
        else:
            q=float(gates(belief.b,masks[idx],alpha_model,method,param,t,[rects[idx]])[0])
        gate_times.append((time.perf_counter_ns()-start)/1000)
        bprior=belief.b.copy();attacker_bprior=attacker.b.copy();attacker_prior_peak=float(attacker.b.max())
        robust_bpriors=np.vstack([m.b.copy() for m in robust.members]) if robust else None
        robust_prior_peaks=np.asarray([m.b.max() for m in robust.members],float) if robust else None
        mask=masks[idx].astype(bool);truth=int(path[t]);task_area=int(mask.sum())
        attacker_positive_floor=disclosure_floor(attacker_bprior,mask)
        eligible=bool(mask[truth])
        if condition.get('availability_by_cell') is not None:
            true_availability=_alpha_vector(condition['availability_by_cell'],side*side)
            available=bool(ext[t,0]<true_availability[truth])
        else:
            available=bool(ext[t,0]<delivery and ext[t,1]<willing and ext[t,2]<on_time)
        q_at_state=float(q[truth]) if channel is not None else q
        reported=bool(eligible and available and ext[t,3]<q_at_state)
        observation=Observation(task.task_id,t,reported)
        belief.observe(task,observation,q)
        attacker.observe(task,observation,q)
        if robust:robust.observe(task,observation,q)
        if method=='bsp' and np.max(belief.b)>max(param,np.max(bprior))+1e-9:
            cap_violations+=1
        elif method=='rbsp':
            if any(m.b.max()>max(float(param),float(p))+1e-9 for m,p in zip(robust.members,robust_prior_peaks)):
                cap_violations+=1
        elif method=='dfbsp':
            c_eff=max(float(positive_cap),float(np.max(bprior))) if reported else max(float(param),float(np.max(bprior)))
            if np.max(belief.b)>c_eff+1e-9:cap_violations+=1
        elif method=='rdfbsp':
            for m,pb in zip(robust.members,robust_bpriors):
                c_eff=max(float(positive_cap),float(pb.max())) if reported else max(float(param),float(pb.max()))
                if m.b.max()>c_eff+1e-9:
                    cap_violations+=1;break
        if method in ('bsp','rbsp','dfbsp','rdfbsp'):
            if attacker.b.max()>max(float(param),attacker_prior_peak)+1e-9:
                attacker_local_cap_violations+=1
            if method in ('dfbsp','rdfbsp'):
                attacker_eff=max(float(positive_cap),attacker_prior_peak) if reported else max(float(param),attacker_prior_peak)
            else:
                attacker_eff=max(float(param),attacker_prior_peak)
            if attacker.b.max()>attacker_eff+1e-9:
                attacker_effective_cap_violations+=1
            if attacker.b.max()>float(param)+1e-9:
                attacker_absolute_cap_violations+=1
        metrics=posterior_metrics(attacker.b,truth,side)
        for k,v in metrics.items():sums[k]+=v
        bn=min(9,int(metrics['peak']*10));bin_counts[bn]+=1;bin_hits[bn]+=metrics['hit'];bin_peaks[bn]+=metrics['peak']
        if condition['scenario']=='geolife':
            ties=np.flatnonzero(np.isclose(attacker.b,attacker.b.max(),rtol=1e-10,atol=1e-14))
            xy=np.array(np.unravel_index(ties,(side,side))).T-np.array(divmod(truth,side))
            error_m+=float(np.linalg.norm(xy*np.array([.3*111320/side,.4*111320*np.cos(np.deg2rad(39.95))/side]),axis=1).mean())
        else:error_m+=metrics['error_cells']*1000
        prior_hit+=posterior_metrics(client_background,truth,side)['hit']
        if is_legit:
            legit+=1
            area_bin='small' if task_area<=8 else ('medium' if task_area<=31 else 'large')
            weight=1.0/task_area
            if eligible and available:
                opportunities+=1;op_cells.add(truth);weighted_opportunities+=weight;area_opp[area_bin]+=1
            if reported:
                complete+=1;done_cells.add(truth);weighted_complete+=weight;area_done[area_bin]+=1
        reports+=reported
        if log:
            rows.append(dict(slot=t,task_id=task.task_id,rectangle=rects[idx],legitimate=is_legit,
                             reported=reported,q=q.tolist() if channel is not None else q,alpha_model=alpha_model,prior_peak=float(bprior.max()),
                             attacker_prior_peak=attacker_prior_peak,attacker_positive_floor=attacker_positive_floor,
                             belief=attacker.b.tolist(),client_peak=float(belief.b.max()),
                             robust_peak_max=float(max(m.b.max() for m in robust.members)) if robust else None,
                             task_area=task_area,positive_cap=positive_cap,
                             evaluator_truth=truth,eligible=eligible,available=available,**metrics))
    out={k:v/slots for k,v in sums.items()}
    out.update(ece=float(np.abs(bin_hits-bin_peaks).sum()/slots),error_m=error_m/slots,seed=seed,user=str(user),scenario=condition['scenario'],condition=condition['id'],
               method=method,param=param,attack=attack,side=side,prior_hit=prior_hit/slots,
               legitimate=legit,opportunities=opportunities,complete=complete,reports=reports,
               utility=complete/opportunities if opportunities else float('nan'),
               weighted_utility=weighted_complete/weighted_opportunities if weighted_opportunities else float('nan'),
               weighted_opportunities=weighted_opportunities,weighted_complete=weighted_complete,
               small_opportunities=area_opp['small'],small_complete=area_done['small'],
               medium_opportunities=area_opp['medium'],medium_complete=area_done['medium'],
               large_opportunities=area_opp['large'],large_complete=area_done['large'],
               raw_completion=complete/legit if legit else float('nan'),coverage=len(done_cells)/len(op_cells) if op_cells else float('nan'),
               reward=complete,
               gate_us_p50=float(np.median(gate_times)),gate_us_p95=float(np.quantile(gate_times,.95)),
               cap_violations=cap_violations,attacker_local_cap_violations=attacker_local_cap_violations,
               attacker_effective_cap_violations=attacker_effective_cap_violations,
               attacker_absolute_cap_violations=attacker_absolute_cap_violations,
               positive_cap=positive_cap,robust_model_count=len(robust.members) if robust else 0,
               alpha_true=alpha_true,alpha_model=alpha_model,move_model=move_model)
    return out,rows

def run(config_path):
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output'])
    if out.exists() and (out/'rows.csv').exists():raise FileExistsError('immutable run already exists: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}),indent=2))
    allrows=[];failures=[];start=time.perf_counter();cache={}
    for ci,cond in enumerate(cfg['conditions']):
        side=cond.get('side',cfg['side']);slots=cfg['slots']
        if side not in cache:cache[side]=candidates(side)
        rects,masks=cache[side]
        if cond['scenario']=='geolife':
            data=load_effective_geolife(cfg['geolife_file'])
            participants=[(int(data['users'][j]),int(data['users'][j]),data['paths'][j][:slots]) for j in range(len(data['users']))]
        else:
            true_move=cond.get('move_true_schedule',cond['move_true'])
            participants=[(seed,u,path_for(seed,u,side,slots,cond['scenario'],true_move)) for seed in cfg['seeds'] for u in range(cfg['users_per_seed'])]
        with gzip.open(out/(cond['id']+'.jsonl.gz'),'wt') as f:
            for j,(seed,user,path) in enumerate(participants):
                try:
                    row,events=simulate(cond,seed,user,path,rects,masks,slots,side,True)
                    allrows.append(row)
                    f.write(json.dumps(dict(seed=seed,user=str(user),events=events))+'\n')
                except Exception as e:
                    failures.append(dict(condition=cond['id'],seed=seed,user=user,error=repr(e)))
                    (out/'failures.json').write_text(json.dumps(failures,indent=2))
                    raise
        print(f'{ci+1}/{len(cfg["conditions"])} {cond["id"]}: n={len(participants)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    with (out/'rows.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    (out/'duration.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start}))
    return allrows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();run(args.config)
