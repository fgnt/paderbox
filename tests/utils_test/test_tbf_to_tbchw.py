import unittest
import numpy as np

from nt.utils.numpy_utils import tbf_to_tbchw

class TestTBFtoTBCHW(unittest.TestCase):

    def setUp(self):
        self.data = np.random.uniform(0, 1, (30, 2, 5))

    def test_shape(self):
        x = tbf_to_tbchw(self.data, 3, 1)
        self.assertEqual(x.shape, (30, 2, 1, 5, 3))
