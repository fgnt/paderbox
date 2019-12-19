import unittest
import numpy as np

from paderbox.array import tbf_to_tbchw

class TestTBFtoTBCHW(unittest.TestCase):

    def setUp(self):
        self.data = np.random.uniform(0, 1, (30, 2, 5))

    def test_shape_even_context(self):
        x = tbf_to_tbchw(self.data, 3, 3, 1)
        self.assertEqual(x.shape, (30, 2, 1, 5, 7))

    def test_shape_left_context(self):
        x = tbf_to_tbchw(self.data, 3, 0, 1)
        self.assertEqual(x.shape, (30, 2, 1, 5, 4))

    def test_shape_right_context(self):
        x = tbf_to_tbchw(self.data, 0, 3, 1)
        self.assertEqual(x.shape, (30, 2, 1, 5, 4))
