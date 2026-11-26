import unittest
import numpy as np
from noise import kraus,noisy,g

class NoiseTests(unittest.TestCase):
    def test_channels(self):
        for channel in ("amplitude","phase","depolarizing"):
            for p in (0,.01,.5,1):
                ks=kraus(channel,p)
                np.testing.assert_allclose(sum(k.conj().T@k for k in ks),np.eye(2),atol=1e-12)
                g.features(noisy(g.GHZ,channel,p))
            np.testing.assert_allclose(noisy(g.GHZ,channel,0),g.GHZ,atol=1e-12)
        np.testing.assert_allclose(noisy(g.GHZ,"amplitude",1),g.projector([1,0,0,0,0,0,0,0]),atol=1e-12)

if __name__=="__main__":
    unittest.main()
