import unittest

import numpy as np
import scipy
import nt.evaluation.sxr as sxr
import nt.speech_enhancement.noise as noise
import nt.testing as tc
from nt.speech_enhancement.noise import get_snr
from nt.speech_enhancement.noise import set_snr
from nt.math.vector import sph2cart
import nt.transform as transform
from nt.speech_enhancement.noise.spherical_habets import _mycohere,_sinf_3D
from math import pi
from numpy.linalg import norm


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
    n_gen_cls = noise.NoiseGeneratorWhite
    n_gen_cls_args = list()
    slope_dB_expected = 0
    slope_dB_atol = .01

    def setUp(self):
        m_gen = self.n_gen_cls(*self.n_gen_cls_args)

    @tc.retry(3)
    def test_single_channel(self):
        time_signal = np.random.randn(1000)
        n = self.n_gen.get_noise_for_signal(time_signal, snr=20)
        tc.assert_equal(n.shape, (1000,))

        SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None])
        tc.assert_almost_equal(SNR, 20, decimal=4)

    @tc.retry(3)
    def test_multi_channel(self):
        time_signal = np.random.randn(3, 1000)
        n = self.n_gen.get_noise_for_signal(time_signal, snr=20)
        tc.assert_equal(n.shape, (3, 1000))

        SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n)
        tc.assert_almost_equal(SNR, 20, decimal=6)

    @tc.retry(5)
    def test_slope(self):
        time_signal = np.random.randn(3, 16000)
        n = self.n_gen.get_noise_for_signal(time_signal, snr=20)
        N = transform.stft(n)
        power_spec = 10*np.log10(noise.get_energy(N, axis=(0, 1)))
        # slope_dB = power_spec[10]-power_spec[100]
        slope_dB, _, _, _, _ = scipy.stats.linregress(10*np.log10(range(1, len(power_spec))), power_spec[1:])
        print('slope_dB: ', slope_dB)
        # from nt.visualization import plot
        # plot.line(10*np.log10(range(1, len(power_spec))), power_spec[1:])

        tc.assert_allclose(slope_dB, self.slope_dB_expected, atol=self.slope_dB_atol)


class TestNoiseGeneratorPink(TestNoiseGeneratorWhite):
    n_gen_cls = noise.NoiseGeneratorPink
    slope_dB_atol = .1
    slope_dB_expected = -1


class TestNoiseGeneratorNoisex92(TestNoiseGeneratorWhite):
    n_gen_cls = noise.NoiseGeneratorNoisex92
    n_gen_cls_args = ['destroyerengine']

    def test_multi_channel(self):
        pass  # currently only single channel supported

    def test_slope(self):
        pass  # no consistent slope to test for


class TestNoiseGeneratorSpherical(TestNoiseGeneratorWhite):
    x1, y1, z1 = sph2cart(0, 0, 0.1)  # Sensor position 1
    x2, y2, z2 = sph2cart(0, 0, 0.2)  # Sensor position 2
    P = np.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]])  # Construct position matrix
    n_gen_cls = noise.NoiseGeneratorSpherical
    n_gen_cls_args = [P]
    slope_dB_expected = 0
    slope_dB_atol = 0.7  # ToDo: analyse, why atol is so high

    def test_single_channel(self):
        pass  # makes no sense

    # @tc.retry(5)
    # def test_slope(self):
    #     # test_spatial_coherences delivers more relevnt solutions
    #     time_signal = np.random.randn(16000,  3)
    #     n = self.n_gen.get_noise_for_signal(time_signal, 20)
    #     N = transform.stft(n)
    #     power_spec = 10*np.log10(noise.get_power(N, axis=(0, 2)))
    #     slope_dB = power_spec[10]-power_spec[100]
    #     tc.assert_allclose(slope_dB, 0, atol=4)

    @tc.retry(3)
    def test_spatial_coherences(self):
        M = 3  # Number of sensors
        NFFT = 256  # Number of frequency bins (for analysis)
        fs = 8000  # Sample frequency
        L = 2**18  # Data length
        c = 340  # Speed of sound
        w = 2*pi*fs*(np.arange(0, NFFT//2+1))/NFFT
        d = norm(self.P-self.P[:, 0], 2, axis=0)  # Sensor distances w.r.t. sensor 1
        z = _sinf_3D(self.P, L, sample_rate=fs)
        sc_sim = np.zeros((M-1, NFFT // 2 + 1))
        sc_theory = np.zeros((M-1, NFFT // 2 + 1))
        for m in range(M-1):  # for m = 1:M-1
            sc, F = _mycohere(z[0, :].T, z[m+1, :].T, NFFT, fs, np.hanning(NFFT), 0.75*NFFT)
            sc_sim[m, :] = np.real(sc)
            sc_theory[m, :] = np.sinc(w*d[m+1]/c/pi)
        delta = [sc_sim[1, :], sc_sim[1, :]]
        max_delta = 0
        for m in range(M-1):
            delta[m] = abs(sc_sim[m, :] - sc_theory[m, :])
            max_delta = max(max(delta[m]), max_delta)
        tc.assert_allclose(max_delta, 0, atol=0.1)
