import unittest
import numpy as np
from src.model import *
from src.experiment import candidates,path_for,simulate

class ChannelTests(unittest.TestCase):
    def test_silence_is_not_false(self):
        b=np.array([.5,.5]);v=update(b,[1,0],.5,1,False)
        np.testing.assert_allclose(v,[1/3,2/3])
    def test_success_independent_of_thinning(self):
        b=np.array([.2,.3,.5]);m=[1,1,0]
        np.testing.assert_allclose(update(b,m,.7,.1,True),update(b,m,.7,1,True))
    def test_zero_gate_and_zero_delivery(self):
        b=np.array([.3,.7])
        np.testing.assert_allclose(update(b,[1,0],.5,0,False),b)
        np.testing.assert_allclose(update(b,[1,0],0,1,False),b)
    def test_transition_mass_and_peak(self):
        b=np.random.default_rng(1).dirichlet(np.ones(64))
        for m in [0,.5,1]:
            v=predict(b,8,m);self.assertAlmostEqual(v.sum(),1);self.assertLessEqual(v.max(),b.max()+1e-12)
    def test_branch_bound_and_maximality(self):
        rng=np.random.default_rng(11)
        for _ in range(500):
            b=rng.dirichlet(np.ones(16)*3);mask=rng.random(16)<.5;alpha=float(rng.uniform(.1,.95));rho=float(rng.uniform(.08,.4));cap=max(rho,b.max())
            q=gates(b,mask,alpha,'bsp',rho,0)[0]
            for r in [True,False]:
                pr=q*alpha*np.dot(b,mask);pr=pr if r else 1-pr
                if pr>1e-12:self.assertLessEqual(update(b,mask,alpha,q,r).max(),cap+1e-9)
            if q<.999 and np.dot(b,mask)>0:
                q2=min(1,q+1e-5)
                self.assertTrue(any(update(b,mask,alpha,q2,r).max()>cap-1e-10 for r in [True,False]))
    def test_positive_only_can_leak_silence(self):
        b=np.array([.2,.2,.2,.4]);m=[1,1,1,0]
        self.assertEqual(gates(b,m,.9,'positive_only',.4,0)[0],1)
        self.assertGreater(update(b,m,.9,1,False).max(),.4)
        self.assertAlmostEqual(gates(b,m,.9,'bsp',.4,0)[0],0)
    def test_boundaries_duplicates_expiry(self):
        t=Task('a',0,0,(0,1,0,1));self.assertEqual(t.mask(2).sum(),1)
        b=PublicBelief(2,0,.5);o=Observation('a',0,False);b.observe(t,o,1)
        with self.assertRaises(ValueError):b.observe(t,o,1)
        with self.assertRaises(ValueError):Task('b',0,0,(0,3,0,1)).mask(2)
        with self.assertRaises(ValueError):PublicBelief(2,0,.5).observe(Task('b',1,0,(0,1,0,1)),Observation('b',1,False),1)
    def test_prior_tie_unbiased(self):
        for x in range(16):self.assertAlmostEqual(posterior_metrics(np.full(16,1/16),x,4)['hit'],1/16)
    def test_public_state_no_ground_truth_fields(self):
        self.assertEqual(set(Observation.__dataclass_fields__),{'task_id','slot','reported'})
        self.assertNotIn('truth',PublicBelief(2,0,.5).__dict__)
    def test_seed_reproducibility(self):
        np.testing.assert_array_equal(path_for(1,2,8,64,'walk',.3),path_for(1,2,8,64,'walk',.3))
    def test_robust_bsp_bounds_intersection_and_maximality(self):
        rng=np.random.default_rng(826)
        for _ in range(200):
            beliefs=np.vstack([rng.dirichlet(np.ones(16)*3) for _ in range(4)])
            alphas=rng.uniform(.2,.95,size=4);mask=rng.random(16)<.55;rho=float(rng.uniform(.08,.35))
            member_q=np.array([gates(b,mask,a,'bsp',rho,0)[0] for b,a in zip(beliefs,alphas)])
            q=robust_bsp_gates(beliefs,mask,alphas,rho)[0]
            self.assertAlmostEqual(q,float(member_q.min()),places=12)
            for b,a in zip(beliefs,alphas):
                cap=max(rho,float(b.max()))
                for reported in [True,False]:
                    pr=q*a*np.dot(b,mask);pr=pr if reported else 1-pr
                    if pr>1e-12:self.assertLessEqual(update(b,mask,a,q,reported).max(),cap+1e-9)
            if q<.999:
                q2=min(1.,q+1e-5);violated=False
                for b,a in zip(beliefs,alphas):
                    cap=max(rho,float(b.max()))
                    for reported in [True,False]:
                        pr=q2*a*np.dot(b,mask);pr=pr if reported else 1-pr
                        if pr>1e-12 and update(b,mask,a,q2,reported).max()>cap-1e-10:
                            violated=True
                self.assertTrue(violated)
    def test_rbsp_contains_informed_attacker_model(self):
        models=[dict(move=m,alpha=a) for m in [.05,.3,.8] for a in [.3,.648,.9]]
        c=dict(delivery=.9,willing=.8,on_time=.9,move_true=.3,move_model=.05,side=8,
               scenario='walk',attack='adaptive',method='rbsp',param=.1,id='rbsp_unit',
               alpha_model=.3,attacker_move=.3,attacker_alpha=.648,robust_models=models)
        r,m=candidates(8);path=path_for(826,0,8,48,'walk',.3)
        out,_=simulate(c,826,0,path,r,m,48,8,False)
        self.assertEqual(out['robust_model_count'],9)
        self.assertEqual(out['cap_violations'],0)
        self.assertEqual(out['attacker_local_cap_violations'],0)
        self.assertEqual(out['attacker_absolute_cap_violations'],0)
    def test_df_bsp_reduces_to_bsp_at_same_cap(self):
        rng=np.random.default_rng(927)
        for _ in range(100):
            b=rng.dirichlet(np.ones(16)*2);mask=rng.random(16)<.6
            rho=.25
            q0=gates(b,mask,.648,'bsp',rho,0)[0]
            q1=df_bsp_gates(b,mask,.648,rho,rho)[0]
            self.assertAlmostEqual(q0,q1,places=12)
    def test_df_bsp_admits_only_declared_positive_exposure(self):
        b=np.full(16,1/16);small=np.zeros(16);small[:4]=1
        self.assertAlmostEqual(disclosure_floor(b,small),.25)
        self.assertEqual(df_bsp_gates(b,small,.648,.1,.2)[0],0)
        q=df_bsp_gates(b,small,.648,.1,.25)[0]
        self.assertGreater(q,0)
        ppos=q*.648*np.dot(b,small)
        if ppos>1e-12:self.assertLessEqual(update(b,small,.648,q,True).max(),.25+1e-9)
        psil=1-ppos
        if psil>1e-12:self.assertLessEqual(update(b,small,.648,q,False).max(),.1+1e-9)
    def test_robust_df_is_memberwise_minimum(self):
        beliefs=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1]])
        masks=np.array([[1,1,1,0],[0,1,1,1]],float);alphas=[.5,.8]
        member=np.vstack([df_bsp_gates(b,masks,a,.2,.5) for b,a in zip(beliefs,alphas)])
        np.testing.assert_allclose(robust_df_bsp_gates(beliefs,masks,alphas,.2,.5),member.min(axis=0))
    def test_df_bsp_random_branch_bounds_and_maximality(self):
        rng=np.random.default_rng(92727)
        for _ in range(200):
            b=rng.dirichlet(np.ones(16)*2);mask=(rng.random(16)<.5).astype(float)
            if not mask.any():mask[int(rng.integers(16))]=1
            alpha=float(rng.uniform(.2,.95));rho=float(rng.uniform(.06,.3));tau=float(rng.uniform(rho,1))
            q=float(df_bsp_gates(b,mask,alpha,rho,tau)[0])
            prior=float(b.max());mass=float(np.dot(b,mask))
            pos=disclosure_floor(b,mask)
            if q>1e-12 and mass>0:
                self.assertLessEqual(pos,max(tau,prior)+1e-9)
            psil=1-q*alpha*mass
            if psil>1e-12:
                self.assertLessEqual(update(b,mask,alpha,q,False).max(),max(rho,prior)+1e-9)
            if q<.999999:
                q2=min(1.,q+1e-5)
                positive_bad=(q2>0 and mass>0 and pos>max(tau,prior)+1e-10)
                silence_bad=False
                psil2=1-q2*alpha*mass
                if psil2>1e-12:
                    silence_bad=update(b,mask,alpha,q2,False).max()>max(rho,prior)-1e-10
                self.assertTrue(positive_bad or silence_bad)
    def test_reset_preserves_distinct_attacker_prior(self):
        client=np.full(4,.25)
        attacker=np.array([.7,.1,.1,.1])
        c=dict(delivery=1.,willing=1.,on_time=1.,move_true=0.,move_model=0.,side=2,
               scenario='static',attack='fixed',method='none',param=1.,id='reset_prior',
               client_prior=client.tolist(),attacker_prior=attacker.tolist(),reset_every=1,
               legitimate_fraction=0.)
        rects,masks=candidates(2);path=np.zeros(2,dtype=int)
        _,events=simulate(c,7,0,path,rects,masks,2,2,True)
        self.assertAlmostEqual(events[0]['attacker_prior_peak'],.7)
        self.assertAlmostEqual(events[1]['attacker_prior_peak'],.7)

    def test_location_dependent_availability_attacker_model(self):
        alpha=[.2,.2,.9,.9]
        c=dict(delivery=1.,willing=1.,on_time=1.,move_true=0.,move_model=0.,side=2,
               scenario='static',attack='adaptive',method='bsp',param=.6,id='loc_alpha',
               alpha_model=.55,attacker_alpha_by_cell=alpha,availability_by_cell=alpha,
               legitimate_fraction=0.)
        rects,masks=candidates(2);path=np.array([0,1,2,3])
        out,_=simulate(c,9,0,path,rects,masks,4,2,False)
        self.assertTrue(np.isfinite(out['hit']))
        self.assertEqual(out['side'],2)

    def test_nonstationary_path_and_two_step_probe_are_reproducible(self):
        sched=[0.,1.,0.,1.,0.,1.]
        p1=path_for(11,0,4,6,'walk',sched);p2=path_for(11,0,4,6,'walk',sched)
        np.testing.assert_array_equal(p1,p2)
        c=dict(delivery=.9,willing=.8,on_time=.9,move_true=.3,move_model=.3,side=4,
               scenario='walk',attack='lookahead2',method='bsp',param=.2,id='lookahead',
               legitimate_fraction=0.,lookahead_beam=6)
        rects,masks=candidates(4)
        out1,e1=simulate(c,12,0,path_for(12,0,4,5,'walk',.3),rects,masks,5,4,True)
        out2,e2=simulate(c,12,0,path_for(12,0,4,5,'walk',.3),rects,masks,5,4,True)
        self.assertEqual([x['rectangle'] for x in e1],[x['rectangle'] for x in e2])
        self.assertAlmostEqual(out1['hit'],out2['hit'])
if __name__=='__main__':unittest.main()
