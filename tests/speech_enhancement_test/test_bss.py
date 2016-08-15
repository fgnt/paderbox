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
