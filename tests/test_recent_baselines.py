import ast, contextlib, io, time, unittest
from pathlib import Path
import numpy as np
from src.recent_baselines import ba,gibu,distances,pml_gates,pml_lp
from src.model import update

class RecentBaselines(unittest.TestCase):
    def test_author_code_conformance(self):
        # Execute only inspected pure algorithm functions, not upstream data I/O.
        source=Path('literature/recent/privic_functions.py').read_text()
        tree=ast.parse(source);names=['BlahutArimotoParis','IBU']
        subset=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[])
        env=dict(np=np,time=time,N_X_paris=9,PosToValParis=lambda x:divmod(x,3))
        exec(compile(subset,'author_functions','exec'),env)
        rng=np.random.default_rng(784);p=rng.dirichlet(np.ones(9));start=np.full((9,9),1/9)
        with contextlib.redirect_stdout(io.StringIO()):
            ref=env['BlahutArimotoParis'](start,p,.7,8)
            ours,_=ba(p,distances(3),.7,fixed_steps=8)
            np.testing.assert_allclose(ours,ref,rtol=1e-13,atol=1e-14)
            q=rng.dirichlet(np.ones(9));init=np.full(9,1/9)
            refp=env['IBU'](init,q,ref,10,range(9),range(9))[1]
            ourp,_=gibu([ref],[q],init,fixed_steps=10)
            np.testing.assert_allclose(ourp,refp,rtol=1e-13,atol=1e-14)

    def test_pml_closed_form_matches_general_lp(self):
        rng=np.random.default_rng(442)
        for _ in range(100):
            p=rng.dirichlet(np.ones(8));mask=rng.random(8)<.6
            if rng.random()<.2:p[rng.integers(8)]=0;p/=p.sum()
            a=rng.uniform(.01,1);eps=rng.uniform(0,4)
            q=pml_gates(p,mask,a,eps)[0];l=a*q*mask;m=p@l
            self.assertAlmostEqual(m,pml_lp(p,mask,a,eps),places=8)
            for branch in [l,1-l]:
                prob=p@branch
                if prob>1e-12:self.assertLessEqual(np.max(branch[p>0])/prob,np.exp(eps)+1e-8)

    def test_pml_zero_and_full_region(self):
        p=np.array([.1,.2,.3,.4])
        self.assertEqual(pml_gates(p,[1,0,0,0],.9,0)[0],0)
        self.assertEqual(pml_gates(p,[1,1,1,1],.9,0)[0],1)
        self.assertAlmostEqual(pml_lp(p,[1,1,1,1],.9,0),.9)

    def test_privic_vector_channel_updates(self):
        c,_=ba(np.full(4,.25),distances(2),1)
        b=np.array([.1,.2,.3,.4]);mask=np.array([1,1,0,0]);q=c@mask
        for reported in [True,False]:
            p=update(b,mask,.7,q,reported)
            likelihood=.7*mask*q
            expected=b*(likelihood if reported else 1-likelihood);expected/=expected.sum()
            np.testing.assert_allclose(p,expected)
        self.assertEqual(np.sum(update(b,mask,.7,q,True)[2:]),0)

    def test_ba_and_gibu_objectives(self):
        p=np.array([.1,.2,.3,.4]);d=distances(2);c,info=ba(p,d,.7)
        self.assertTrue(info['converged']);self.assertLessEqual(info['objective'],.7*np.mean(p@d)+1e-8)
        qs=[np.array([4.,3.,2.,1.]),np.array([1.,2.,3.,4.])];cs=[c,.5*c+.125]
        before=sum(np.dot(q,np.log(np.full(4,.25)@a)) for q,a in zip(qs,cs))
        est,diagnostic=gibu(cs,qs,np.full(4,.25))
        self.assertGreaterEqual(diagnostic['log_likelihood'],before-1e-9)
        self.assertAlmostEqual(est.sum(),1)

    def test_policy_envelope_matches_mixture_lp(self):
        from src.analyze_recent import lower_hull
        from scipy.optimize import linprog
        rng=np.random.default_rng(985)
        for _ in range(25):
            x=np.r_[0,rng.random(8),1];y=rng.random(10);target=rng.uniform(.05,.95)
            hull=lower_hull(zip(x,y));v=np.interp(target,[p[0] for p in hull],[p[1] for p in hull])
            lp=linprog(y,A_eq=np.array([np.ones(10),x]),b_eq=[1,target],bounds=(0,1),method='highs')
            self.assertTrue(lp.success);self.assertAlmostEqual(v,lp.fun,places=10)

if __name__=='__main__':unittest.main()
