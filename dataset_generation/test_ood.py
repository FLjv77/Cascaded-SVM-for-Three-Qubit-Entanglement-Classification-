"""Regression checks for UPB normalization and physical validity."""
import unittest
import numpy as np
from generate import features, rotate
from generate_ood import upb, ppt_min

class UPBTests(unittest.TestCase):
    def test_normalized_rank_four_projector(self):
        rho = upb()
        np.testing.assert_allclose(np.trace(rho), 1, atol=1e-12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
        np.testing.assert_allclose(np.linalg.eigvalsh(rho), [0]*4+[.25]*4, atol=1e-12)
        np.testing.assert_allclose(rho @ rho, rho/4, atol=1e-12)
        self.assertEqual(features(rho).shape, (63,))
        for cut in range(3):
            self.assertGreaterEqual(ppt_min(rho,cut), -1e-12)

    def test_local_rotations_preserve_normalization(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            rho = rotate(rng, upb())
            np.testing.assert_allclose(np.trace(rho), 1, atol=1e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh(rho), [0]*4+[.25]*4, atol=1e-12)
            self.assertEqual(features(rho).shape, (63,))

if __name__ == "__main__":
    unittest.main()
