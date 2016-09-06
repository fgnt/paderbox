import unittest
import numpy as np
from nt.speech_enhancement import bss


def _random_stft(*shape):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


class TestNormalizeObservation(unittest.TestCase):
    def test_forbidden_list_input(self):
        with self.assertRaises(ValueError):
            bss.normalize_observation(
                _random_stft(2, 513, 4),
                frequency_norm=True,
                max_sensor_distance=0.2
            )


class TestSetSNR(unittest.TestCase):
    def test_time_domain(self):
        K, D, T = 2, 3, 50
        snr = 13.
        x = np.random.normal(size=(K, D, T))
        n = np.random.normal(size=(1, D, T))
        bss.set_snr(x, n, snr)

        np.testing.assert_almost_equal(np.mean(x ** 2), 1)
        np.testing.assert_almost_equal(np.mean(x[0, 0] ** 2), 1)

        np.testing.assert_almost_equal(
            10. * np.log10(np.mean(x ** 2) / np.mean(n ** 2)),
            snr
        )
