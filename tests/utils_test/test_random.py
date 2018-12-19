import unittest
import numpy as np
import paderbox.testing as tc
import paderbox.utils.random_utils as random


class TestRandom(unittest.TestCase):
    def test_uniform_scalar(self):
        scalar = random.uniform()
        tc.assert_equal(scalar.shape, (1,))

    def test_hermitian_2D(self):
        shape = (2, 2)
        matrix = random.hermitian(*shape)
        tc.assert_equal(matrix.shape, shape)
        tc.assert_equal(matrix, np.swapaxes(matrix, -1, -2).conj())

    def test_hermitian_many_dimensions(self):
        shape = (3, 4, 5, 2, 2)
        matrix = random.hermitian(*shape)
        tc.assert_equal(matrix.shape, shape)
        tc.assert_equal(matrix, np.swapaxes(matrix, -1, -2).conj())
