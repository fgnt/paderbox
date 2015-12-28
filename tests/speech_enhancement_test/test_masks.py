import unittest
from nt.speech_enhancement.mask_estimation import simple_ideal_soft_mask
import nt.testing as tc
import numpy as np

F, T, D, K = 51, 31, 6, 2
X_all = np.random.rand(F, T, D, K) + 1j * np.random.rand(F, T, D, K)
X, N = (X_all[:, :, :, 0], X_all[:, :, :, 1])

class SimpleIdealSoftMaskTests(unittest.TestCase):

    def test_single_input(self):
        M1 = simple_ideal_soft_mask(X_all)
        tc.assert_equal(M1.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M1, axis=2), 1)
        return M1

    def test_separate_input(self):
        M2 = simple_ideal_soft_mask(X, N)
        tc.assert_equal(M2.shape, (51, 31, 2))
        tc.assert_almost_equal(np.sum(M2, axis=2), 1)
        return M2

    def test_separate_input_equals_single_input(self):
        tc.assert_equal(self.test_single_input(), self.test_separate_input())

    def test_(self):
        M3 = simple_ideal_soft_mask(X_all, N)
        tc.assert_equal(M3.shape, (51, 31, 3))
        tc.assert_almost_equal(np.sum(M3, axis=2), 1)

    def test_negative_feature_bin(self):
        M4 = simple_ideal_soft_mask(X, N, feature_dim=-3)
        tc.assert_equal(M4.shape, (51, 6, 2))
        tc.assert_almost_equal(np.sum(M4, axis=2), 1)
