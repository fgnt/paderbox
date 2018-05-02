import unittest

import numpy as np
from nt.reverb.reverb_utils import convolve
import nt.testing as tc


class TestConvolution(unittest.TestCase):
    def test_one_dimension(self):
        signal = np.asarray([1, 2, 3])
        impulse_response = np.asarray([1, 1])
        x = convolve(signal, impulse_response).tolist()
        tc.assert_allclose(x, [1, 3, 5, 3])

    def test_kt_kdl(self):
        K, T, D, filter_length = 2, 12, 3, 5
        signal = np.random.normal(size=(K, T))
        impulse_response = np.random.normal(size=(K, D, filter_length))
        x = convolve(signal, impulse_response)
        tc.assert_allclose(x.shape, [K, D, T + filter_length - 1])

    def test_t_dl(self):
        K, T, D, filter_length = 2, 12, 3, 5
        signal = np.random.normal(size=(T,))
        impulse_response = np.random.normal(size=(D, filter_length))
        x = convolve(signal, impulse_response)
        tc.assert_allclose(x.shape, [D, T + filter_length - 1])


if __name__ == '__main__':
    unittest.main()
