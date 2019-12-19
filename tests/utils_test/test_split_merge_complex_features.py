import unittest
import numpy as np
from paderbox.array import split_complex_features, merge_complex_features

T, B, F = 400, 6, 513
A = np.random.uniform(size=(T, B, F)) + 1j * np.random.uniform(size=(T, B, F))


class TestSplitMerge(unittest.TestCase):
    def test_identity_operation(self):
        splitted = split_complex_features(A)
        assert splitted.shape == (T, B, 2*F)
        merged = merge_complex_features(splitted)
        np.testing.assert_almost_equal(A, merged)

    def test_split_toy_example(self):
        A = np.asarray([[[1 + 2j]]])
        splitted = split_complex_features(A)
        np.testing.assert_almost_equal(splitted, np.asarray([[[1, 2]]]))

    def test_merge_toy_example(self):
        A = np.asarray([[[1, 2]]])
        merged = merge_complex_features(A)
        np.testing.assert_almost_equal(merged, np.asarray([[[1 + 2j]]]))
