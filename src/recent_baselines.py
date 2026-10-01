"""2024 mechanisms and explicitly identified truthful-task adapters.

PRIVIC: Biswas and Palamidessi, PoPETs 2024(1), Algorithms 1--3.
PML constraints: Grosse et al., TIFS 19 (2024), Lemma 1.
These are not reproductions of published crowdsensing deployments.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from .model import normalize
from .geolife import load_effective_geolife

def distances(side):
    xy=np.array(np.unravel_index(np.arange(side*side),(side,side))).T
    return np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)

def ba(prior,distance,beta,tol=1e-8,max_iter=100000,fixed_steps=None):
    """Vectorized version of the author BA recurrence, uniform initial channel."""
    p=normalize(prior);n=len(p);c=np.full(n,1/n);kernel=np.exp(-beta*distance)
    old=np.full((n,n),1/n);residual=float('inf')
    for it in range(fixed_steps or max_iter):
        a=kernel*c;channel=a/a.sum(axis=1,keepdims=True)
        residual=float(np.max(np.abs(channel-old)))
        c=p@channel;old=channel
        if fixed_steps is None and residual<tol:break
    out=p@channel
    mi=float(np.sum(p[:,None]*channel*np.log(np.maximum(channel,1e-300)/np.maximum(out,1e-300))))
    distortion=float(np.sum(p[:,None]*channel*distance))
    return channel,dict(iterations=it+1,residual=residual,converged=residual<tol,
                       mi_nats=mi,distortion_cells=distortion,objective=mi+beta*distortion)

def gibu(channels,counts,initial,tol=1e-8,max_iter=100000,fixed_steps=None):
    """Generalized IBU with sample-count weights; a singleton is ordinary IBU."""
    p=normalize(initial);total=sum(c.sum() for c in counts)
    for it in range(fixed_steps or max_iter):
        multiplier=np.zeros_like(p)
        for channel,count in zip(channels,counts):
            den=p@channel
            ratio=np.divide(count,den,out=np.zeros_like(den),where=den>0)
            if np.any((den<=0)&(count>0)):raise ValueError('impossible noisy sample')
            multiplier+=channel@ratio/total
        nxt=normalize(p*multiplier);residual=float(np.max(np.abs(nxt-p)));p=nxt
        if fixed_steps is None and residual<tol:break
    ll=float(sum(np.dot(count,np.log(np.maximum(p@c,1e-300))) for c,count in zip(channels,counts)))
    return p,dict(iterations=it+1,residual=residual,converged=residual<tol,log_likelihood=ll)

def privic_train(samples,side,beta,seed=42024,rounds=8):
    """Paper Algorithm 1 (BA initialization and final GIBU, not script shortcuts).

    Inputs are development locations only. Release a noisy surrogate each round,
    estimate its source using IBU, combine estimates by sample count, then GIBU.
    The resulting channel is frozen for test users.
    """
    rng=np.random.default_rng(seed);samples=rng.permutation(samples)
    n=side*side;theta=np.full(n,1/n);distance=distances(side)
    channels=[];counts=[];logs=[];seen=0
    for batch in np.array_split(samples,rounds):
        c,ba_info=ba(theta,distance,beta)
        cdf=np.cumsum(c[batch],axis=1);cdf[:,-1]=1
        noisy=(rng.random(len(batch))[:,None]>cdf).sum(axis=1)
        count=np.bincount(noisy,minlength=n).astype(float)
        mu,ibu_info=gibu([c],[count],theta)
        theta=(seen*theta+len(batch)*mu)/(seen+len(batch));seen+=len(batch)
        channels.append(c);counts.append(count);logs.append(dict(ba=ba_info,ibu=ibu_info,n=len(batch)))
    estimate,gi_info=gibu(channels,counts,np.full(n,1/n))
    c,ba_info=ba(estimate,distance,beta)
    return c,estimate,dict(rounds=logs,gibu=gi_info,final_ba=ba_info),channels,counts

def pml_gates(b,masks,alpha,epsilon):
    """Exact completion optimum in the truthful binary PML polytope.

    Region mass M < exp(-epsilon) forbids reports. Otherwise max report mass is
    min(alpha*M,1-exp(-epsilon)) when outside has positive prior. Uniform
    thinning attains this bound, even among location-dependent gates.
    """
    if epsilon<0 or not 0<=alpha<=1:raise ValueError('invalid channel parameter')
    mass=np.atleast_2d(masks)@b;cut=np.exp(-epsilon)
    q=np.minimum(1,np.divide(-np.expm1(-epsilon),alpha*mass,out=np.zeros_like(mass),where=mass>0))
    q=np.where(mass+1e-12<cut,0,q)
    q=np.where((mass>=1-1e-14)|(mass<=1e-300),1,q)
    return q

def pml_lp(b,mask,alpha,epsilon):
    """Independent general LP, used for conformance tests, not simulated timing."""
    from scipy.optimize import linprog
    active=np.flatnonzero(b>0);p=b[active];inside=np.asarray(mask)[active]
    k=np.exp(epsilon);n=len(p)
    a=np.eye(n)-k*p[None,:]
    sol=linprog(-p,A_ub=np.vstack([a,-a]),b_ub=np.r_[np.zeros(n),np.full(n,k-1)],
                bounds=[(0,alpha if x else 0) for x in inside],method='highs')
    if not sol.success:raise RuntimeError(sol.message)
    return float(-sol.fun)

def prepare(config_path='configs/recent_baselines.json',output=None):
    from .experiment import path_for
    cfg=json.loads(Path(config_path).read_text());out=Path(output or cfg['training_output'])
    if (out/'manifest.json').exists():raise FileExistsError('training output exists')
    out.mkdir(parents=True,exist_ok=True);side=cfg['side'];manifest=[]
    for scenario in ['static','walk','commute','geolife']:
        if scenario=='geolife':
            data=load_effective_geolife('data/geolife_development.npz');samples=data['paths'].ravel();ids=data['users'].tolist()
        else:
            samples=np.concatenate([path_for(s,u,side,48,scenario,0 if scenario=='static' else .3)
                                    for s in range(4000,4020) for u in range(4)])
            ids=list(range(4000,4020))
        for beta in cfg['privic_betas']:
            start=time.perf_counter();c,theta,info,channels,counts=privic_train(samples,side,beta)
            target=out/f'{scenario}_{beta:g}.npz'
            np.savez_compressed(target,channel=c,estimate=theta,round_channels=np.array(channels),round_counts=np.array(counts))
            record=dict(scenario=scenario,beta=beta,development_ids=ids,n_samples=len(samples),
                        wall_seconds=time.perf_counter()-start,source_sample_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                        channel_file=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),**info)
            manifest.append(record);print(scenario,beta,round(record['wall_seconds'],2),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/recent_baselines.json');ap.add_argument('--output')
    a=ap.parse_args();prepare(a.config,a.output)
