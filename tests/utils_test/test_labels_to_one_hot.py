import unittest
import numpy as np
import nt.testing as tc
from nt.utils.numpy_utils import labels_to_one_hot


class TestLabelsToOneHot(unittest.TestCase):
    def test_curated_input_with_keepdims_true(self):
        labels = np.asarray([
            [0, 1, 2, 0]
        ])
        dtype = np.float32
        expected_mask = np.asarray([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=dtype)
        tc.assert_equal(
            labels_to_one_hot(labels, categories=5, keepdims=True),
            expected_mask
        )

    def test_curated_input_with_keepdims_false(self):
        labels = np.asarray([0, 1, 2, 0])
        dtype = np.float32
        expected_mask = np.asarray([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=dtype)
        tc.assert_equal(
            labels_to_one_hot(labels, categories=5, keepdims=False),
            expected_mask
        )
