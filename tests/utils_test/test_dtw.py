import unittest
from paderbox.utils.dtw import dtw
from numpy.testing import assert_equal
import numpy as np


class TestDTW(unittest.TestCase):
    def test_dtw(self):
        a = np.asarray([1, 1, 2, 2, 3, 4])[:, None]
        b = np.asarray([1.1, 1.2, 1, 2, 3, 4])[:, None]
        dist_min, _, _, path = dtw(a, b, lambda x, y: np.abs(x - y) ** 2)

        res_path = (
            np.asarray([0, 0, 1, 2, 3, 4, 5]),
            np.asarray([0, 1, 2, 3, 3, 4, 5])
        )

        assert_equal(path[0], res_path[0])
        assert_equal(path[1], res_path[1])
        self.assertAlmostEqual(dist_min, 0.049999999999999996)
