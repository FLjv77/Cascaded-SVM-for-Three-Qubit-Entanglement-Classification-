import tempfile
import unittest
from pathlib import Path
import numpy as np
import generate as g
from fixed_datasets import build,verify

class FixedTests(unittest.TestCase):
    def test_mixed(self):
        rng=np.random.default_rng(42)
        for _ in range(12):
            rho=g.mixed_bi(rng)
            self.assertEqual(g.features(rho).shape,(63,))
            self.assertLess(min(g.ppt_min_three(rho,k) for k in range(3)),-1e-8)
    def test_fixed(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/"a"; b=Path(t)/"b"
            build(a,40,42,.7); verify(a,40,42,.7)
            build(b,40,42,.7); verify(b,40,42,.7)
            self.assertEqual((a/"manifest.json").read_bytes(),(b/"manifest.json").read_bytes())

if __name__=="__main__":
    unittest.main()
