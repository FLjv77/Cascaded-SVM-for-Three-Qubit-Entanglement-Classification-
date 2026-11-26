import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import generate as g

class GeneratorTests(unittest.TestCase):
    def test_ppt(self):
        bell = g.projector([1,0,0,1])
        original = bell.copy()
        self.assertAlmostEqual(np.linalg.eigvalsh(g.partial_transpose(bell))[0], -0.5)
        np.testing.assert_array_equal(bell,original)

    def test_permutations(self):
        rho = g.projector([0,1,0,0,0,0,0,0])
        for order in [(0,1,2),(2,1,0),(0,2,1)]:
            expected = [0,0,1]
            index = sum(expected[i]*2**(2-j) for j,i in enumerate(order))
            self.assertAlmostEqual(g.permute(rho,order)[index,index].real,1)

    def test_features(self):
        self.assertEqual(len(g.LABELS),63)
        self.assertEqual(g.LABELS[:9],["IIX","IIY","IIZ","IXI","IYI","IZI","XII","YII","ZII"])
        f = dict(zip(g.LABELS,g.features(g.GHZ)))
        self.assertAlmostEqual(f["XXX"],1)
        self.assertAlmostEqual(f["ZZI"],1)
        self.assertAlmostEqual(f["ZII"],0)
        np.testing.assert_allclose(g.features(np.eye(8)/8),0)

    def test_thresholds(self):
        self.assertAlmostEqual(g.EPSILON_C,0.373149,places=6)
        self.assertEqual(g.ALPHA_C,0.479)
        self.assertGreater(g.ALPHA_MARGIN,0)

    def test_samples(self):
        rng = np.random.default_rng(7)
        for model in g.FILES:
            for label in (0,1):
                for _ in range(12):
                    self.assertEqual(g.features(g.sample(rng,model,label)).shape,(63,))

    def test_csv_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/"a",Path(tmp)/"b"
            g.generate(a,12,42)
            g.generate(b,12,42)
            for name in g.FILES.values():
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
                with (a/name).open() as f:
                    rows=list(csv.reader(f))
                self.assertEqual(rows[0],g.LABELS+["label"])
                self.assertEqual(len(rows),13)
                self.assertTrue(all(len(row)==64 for row in rows))
                self.assertEqual(sum(int(row[-1]) for row in rows[1:]),6)

if __name__=="__main__":
    unittest.main()
