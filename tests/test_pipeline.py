import copy,gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
from src.model import gates,update
from src.experiment import simulate,candidates,path_for
from src.prepare_data import minute_window
from src.geolife import coordinate_fingerprint,state_path_fingerprint,effective_geolife_splits
from src.fit_ambiguity import fit_move_coarsened

class PipelineTests(unittest.TestCase):
    def test_kl_branches_and_maximality(self):
        rng=np.random.default_rng(29)
        for _ in range(100):
            b=rng.dirichlet(np.ones(16));mask=rng.random(16)<.6;alpha=.7;kappa=float(rng.uniform(.05,2))
            q=gates(b,mask,alpha,'kl',kappa,0)[0]
            def divergence(q,r):
                p=update(b,mask,alpha,q,r);nz=p>0
                return np.sum(p[nz]*np.log(p[nz]/b[nz]))
            if q>0:self.assertLessEqual(divergence(q,True),kappa+1e-8)
            self.assertLessEqual(divergence(q,False),kappa+1e-8)
            if q<.999 and np.dot(b,mask)>0:
                self.assertGreater(max(divergence(q+1e-5,r) for r in [True,False]),kappa-1e-8)
    def test_missing_minute_not_interpolated(self):
        def fixture(minutes):
            return '\n'.join(['header']*6+[f'39.9,116.3,0,0,{m/1440:.12f},2008-01-01,00:00:00' for m in minutes])
        self.assertIsNone(minute_window(fixture([1,2,4]),slots=3))
        self.assertIsNotNone(minute_window(fixture([1,2,3]),slots=3))
    def test_data_split_disjoint(self):
        if not all(Path(f'data/geolife_{split}.npz').exists() for split in ('development','validation','test')):
            self.skipTest('GeoLife caches require original GeoLife download and python -m src.prepare_data')
        splits,_=effective_geolife_splits()
        names=['development','validation','test']
        users=[set(map(int,splits[s]['users'])) for s in names]
        states=[{state_path_fingerprint(x) for x in splits[s]['paths']} for s in names]
        for i in range(3):
            for j in range(i):
                self.assertFalse(users[i]&users[j])
                self.assertFalse(states[i]&states[j])
        self.assertTrue(all(len(splits[s]['users'])>0 for s in names))
    def test_state_path_group_keeps_owning_split_members(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);same=np.array([[39.9,116.3],[39.91,116.31]])
            other=np.array([[39.92,116.32],[39.93,116.33]])
            third=np.array([[39.94,116.34],[39.95,116.35]])
            fourth=np.array([[39.96,116.36],[39.97,116.37]])
            np.savez(root/'geolife_development.npz',users=np.array([70,75]),paths=np.array([[1,2],[1,2]]),coords=np.array([same,fourth]))
            np.savez(root/'geolife_validation.npz',users=np.array([11]),paths=np.array([[1,2]]),coords=np.array([other]))
            np.savez(root/'geolife_test.npz',users=np.array([13,88]),paths=np.array([[1,2],[3,4]]),coords=np.array([third,other]))
            splits,dropped=effective_geolife_splits({s:root/f'geolife_{s}.npz' for s in ['development','validation','test']})
            self.assertEqual(splits['development']['users'].tolist(),[70,75])
            self.assertEqual(splits['validation']['users'].tolist(),[])
            self.assertEqual(splits['test']['users'].tolist(),[88])
            self.assertEqual({d['user'] for d in dropped},{11,13})
    def test_coordinate_fingerprint_does_not_collapse_nearby_points(self):
        a=np.array([[39.90000000,116.30000000],[39.91000000,116.31000000]])
        b=a.copy();b[0,0]+=1e-8
        self.assertNotEqual(coordinate_fingerprint(a),coordinate_fingerprint(b))
    def test_shared_normal_tasks_and_exogenous_draws(self):
        c=json.loads(Path('configs/final.json').read_text())['conditions'][0]
        r,m=candidates(8);path=path_for(1000,0,8,48,c['scenario'],c['move_true'])
        _,a=simulate(c,1000,0,path,r,m,48,8,True)
        c=copy.deepcopy(c);c.update(method='bsp',param=.1)
        _,b=simulate(c,1000,0,path,r,m,48,8,True)
        for x,y in zip(a,b):
            self.assertEqual(x['legitimate'],y['legitimate']);self.assertEqual(x['available'],y['available'])
            if x['legitimate']:self.assertEqual(x['rectangle'],y['rectangle'])
    def test_default_replay_matches_archived_numerics(self):
        cfg=json.loads(Path('configs/final.json').read_text());c=next(c for c in cfg['conditions'] if c['scenario']=='walk' and c['method']=='bsp' and c['param']==.1 and c['attack']=='adaptive')
        r,m=candidates(8);path=path_for(1000,0,8,48,'walk',.3)
        _,events=simulate(c,1000,0,path,r,m,48,8,True)
        archive=Path('results/final')/(c['id']+'.jsonl.gz')
        if not archive.exists():
            self.skipTest('Frozen event fixture is archived at the immutable Bpaper source commit; copy it here to run this comparison')
        with gzip.open(archive,'rt') as f:old=json.loads(next(f))['events']
        for a,b in zip(events,old):
            self.assertEqual(a['reported'],b['reported']);self.assertEqual(tuple(a['rectangle']),tuple(b['rectangle']))
            np.testing.assert_allclose(a['belief'],b['belief'],rtol=1e-13,atol=1e-14)
    def test_coarsened_move_fit_recovers_reflecting_walk(self):
        paths=np.vstack([path_for(9100,u,8,240,'walk',.3) for u in range(48)])
        self.assertLess(abs(fit_move_coarsened(paths,8)-.3),.04)
        static=np.vstack([path_for(9200,u,8,80,'static',0) for u in range(8)])
        self.assertEqual(fit_move_coarsened(static,8),0.0)
    def test_identity_reset_preserves_distinct_client_and_attacker_priors(self):
        side=2
        rects,masks=candidates(side)
        client_prior=[.7,.1,.1,.1]
        attacker_prior=[.1,.15,.2,.55]
        c=dict(id='reset_distinct_priors',delivery=.9,willing=.8,on_time=.9,move_model=0.0,method='bsp',param=.9,
               attack='fixed',scenario='static',reset_every=1,client_prior=client_prior,
               attacker_prior=attacker_prior,legitimate_fraction=1.0)
        path=np.array([0,0])
        _,events=simulate(c,9901,0,path,rects,masks,2,side,True)
        for e in events:
            self.assertAlmostEqual(e['prior_peak'],.7)
            self.assertAlmostEqual(e['attacker_prior_peak'],.55)
        self.assertNotEqual(int(np.argmax(client_prior)),int(np.argmax(attacker_prior)))
if __name__=='__main__':unittest.main()
