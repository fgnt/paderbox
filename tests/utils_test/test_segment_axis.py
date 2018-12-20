import unittest

import numpy as np
from numpy.testing import assert_equal

from paderbox.utils.numpy_utils import segment_axis_v2


class TestSegment(unittest.TestCase):
    def test_simple(self):
        assert_equal(segment_axis_v2(np.arange(6), length=3, shift=3),
                     np.array([[0, 1, 2], [3, 4, 5]]))

        assert_equal(segment_axis_v2(np.arange(7), length=3, shift=2),
                     np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))

        assert_equal(segment_axis_v2(np.arange(7), length=3, shift=1),
                     np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
                               [4, 5, 6]]))

        assert_equal(segment_axis_v2(np.arange(7), length=3, shift=-1),
                     [[4, 5, 6], [3, 4, 5], [2, 3, 4], [1, 2, 3], [0, 1, 2]])

    def test_error_checking(self):
        self.assertRaises(ValueError,
                          lambda: segment_axis_v2(np.arange(7), length=0,
                                                  shift=0))
        self.assertRaises(ValueError,
                          lambda: segment_axis_v2(np.arange(7), length=3,
                                                  shift=0))

    def test_ending(self):
        assert_equal(segment_axis_v2(np.arange(6), length=3, shift=2, end='cut'),
                     np.array([[0, 1, 2], [2, 3, 4]]))
        assert_equal(
            segment_axis_v2(
                np.arange(6)+10, length=3, shift=2, end='pad', pad_mode='wrap'),
                [[10, 11, 12], [12, 13, 14], [14, 15, 10]]
        )
        assert_equal(segment_axis_v2(np.arange(6), length=3, shift=2, end='pad',
                                     pad_value=-17),
                     np.array([[0, 1, 2], [2, 3, 4], [4, 5, -17]]))

    def test_multidimensional(self):
        assert_equal(segment_axis_v2(np.ones((2, 3, 4, 5, 6)), axis=3, length=3,
                                  shift=2).shape,
                     (2, 3, 4, 2, 3, 6))

        assert_equal(
            segment_axis_v2(np.ones((2, 3, 4, 5, 6)), axis=2, length=3, shift=2,
                         end='cut').shape,
            (2, 3, 1, 3, 5, 6))

        assert_equal(
            segment_axis_v2(np.ones((2, 3, 4, 5, 6)), axis=2, length=3, shift=2,
                         end='wrap').shape,
            (2, 3, 2, 3, 5, 6))

        assert_equal(
            segment_axis_v2(np.ones((2, 3, 4, 5, 6)), axis=2, length=3, shift=2,
                         end='pad').shape,
            (2, 3, 2, 3, 5, 6))
