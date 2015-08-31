import unittest
import numpy as np
import nt.testing
from scipy import signal

import nt.testing as tc

from nt.speech_enhancement.noise import set_snr
from nt.speech_enhancement.noise import get_snr

class TestNoiseMethods(unittest.TestCase):
    def test_set_and_get_single_source_snr(self):
        T = 100
        F = 123
        X = np.random.normal(size=(T, F)) + 1j * np.random.normal(size=(T, F))
        N = np.random.normal(size=(T, F)) + 1j * np.random.normal(size=(T, F))
        X = 5*X

        snr = 20.0 # dB
        X, N = set_snr(X, N, snr)
        calculated_snr = get_snr(X, N)

        print(calculated_snr)
        print(snr)
        nt.testing.assert_allclose(calculated_snr, snr, atol=1e-3)
