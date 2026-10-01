import unittest
from src.analyze import METRICS,aggregate

class AnalysisEstimatorTests(unittest.TestCase):
    def test_cluster_bootstrap_aggregate_keeps_window_weighted_estimand(self):
        def cluster(value,n,complete,opportunities):
            d={k:float(value) for k in METRICS}
            d.update({f'__n_{k}':int(n) for k in METRICS})
            d.update(__n_rows=int(n),complete=float(complete),opportunities=float(opportunities))
            return d
        small=cluster(.2,1,1,2)
        large=cluster(.8,3,3,6)
        out=aggregate([small,large])
        self.assertAlmostEqual(out['hit'],.65)
        self.assertAlmostEqual(out['peak'],.65)
        self.assertAlmostEqual(out['utility'],.5)

if __name__=='__main__':
    unittest.main()
