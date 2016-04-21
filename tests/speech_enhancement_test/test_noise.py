import unittest

import numpy as np
import nt.evaluation.sxr as sxr
import nt.speech_enhancement.noise as noise
import nt.testing as tc
from nt.speech_enhancement.noise import get_snr
from nt.speech_enhancement.noise import set_snr
from nt.utils.math_ops import sph2cart
import nt.transform as transform


class TestNoiseMethods(unittest.TestCase):
    def test_set_and_get_single_source_snr(self):
        T = 100
        F = 123
        X = np.random.normal(size=(T, F)) + 1j * np.random.normal(size=(T, F))
        N = np.random.normal(size=(T, F)) + 1j * np.random.normal(size=(T, F))
        X *= 5

        snr = 20.0  # dB
        set_snr(X, N, snr)
        calculated_snr = get_snr(X, N)

        print(calculated_snr)
        print(snr)
        tc.assert_allclose(calculated_snr, snr, atol=1e-3)


class TestNoiseGeneratorWhite(unittest.TestCase):
    n_gen = noise.NoiseGeneratorWhite()

    @tc.retry(3)
    def test_single_channel(self):
        time_signal = np.random.randn(1000)
        n = self.n_gen.get_noise_for_signal(time_signal, 20)
        tc.assert_equal(n.shape, (1000,))

        SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None])
        tc.assert_almost_equal(SNR, 20, decimal=6)

    @tc.retry(3)
    def test_multi_channel(self):
        time_signal = np.random.randn(1000, 3)
        n = self.n_gen.get_noise_for_signal(time_signal, 20)
        tc.assert_equal(n.shape, (1000, 3))

        SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n)
        tc.assert_almost_equal(SNR, 20, decimal=6)

    @tc.retry(3)
    def test_slope(self):
        time_signal = np.random.randn(16000,  3)
        n = self.n_gen.get_noise_for_signal(time_signal, 20)
        N = transform.stft(n)
        power_spec = 10*np.log10(noise.get_power(N, axis=(0, 2)))
        slope_dB = power_spec[10]-power_spec[100]
        self.assertAlmostEqual(slope_dB, 0, delta=0.5)


class TestNoiseGeneratorPink(TestNoiseGeneratorWhite):
    n_gen = noise.NoiseGeneratorPink()

    @tc.retry(3)
    def test_slope(self):
        time_signal = np.random.randn(16000, 5)
        n = self.n_gen.get_noise_for_signal(time_signal, 20)
        N = transform.stft(n)
        power_spec = 10*np.log10(noise.get_power(N, axis=(0, 2)))
        slope_dB = power_spec[10]-power_spec[100]
        self.assertAlmostEqual(slope_dB, 10, delta=0.5)


class TestNoiseGeneratorNoisex92(TestNoiseGeneratorWhite):
    n_gen = noise.NoiseGeneratorNoisex92('destroyerengine')

    def test_multi_channel(self):
        pass  # currently only single channel supported

    def test_slope(self):
        pass  # no consistent slope to test for


class TestNoiseGeneratorSpherical(TestNoiseGeneratorWhite):
    x1, y1, z1 = sph2cart(0, 0, 0.1)  # Sensor position 1
    x2, y2, z2 = sph2cart(0, 0, 0.2)  # Sensor position 2
    P = np.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]])  # Construct position matrix
    n_gen = noise.NoiseGeneratorSpherical(P)

    def test_single_channel(self):
        pass  # makes no sense

    @tc.retry(3)
    def test_slope(self):
        time_signal = np.random.randn(16000,  3)
        n = self.n_gen.get_noise_for_signal(time_signal, 20)
        N = transform.stft(n)
        power_spec = 10*np.log10(noise.get_power(N, axis=(0, 2)))
        slope_dB = power_spec[10]-power_spec[100]
        self.assertAlmostEqual(slope_dB, 0, delta=4)
