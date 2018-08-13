import numpy as np
import unittest

from nt.evaluation.sxr import input_sxr, output_sxr
from nt.utils.numpy_utils import morph


class TestSXR(unittest.TestCase):
    def setUp(self):
        samples = 8000
        self.s1 = np.random.normal(size=(samples,))
        self.s2 = np.random.normal(size=(samples,))
        self.n = np.random.normal(size=(samples,))

        self.s1 /= np.sqrt(np.mean(self.s1 ** 2))
        self.s2 /= np.sqrt(np.mean(self.s2 ** 2))
        self.n /= np.sqrt(np.mean(self.n ** 2))

    def test_input_sxr(self):
        images = morph('kt->k1t', 10 * np.stack((self.s1, self.s2)))
        noise = morph('t->1t', self.n)
        sdr, sir, snr = input_sxr(images, noise, average_sources=False)
        assert all(sir == input_sxr(images, noise, average_sources=False).sir)
        np.testing.assert_allclose(sdr, 2 * [10 * np.log10(100/101)], atol=1e-6)
        np.testing.assert_allclose(sir, 2 * [0], atol=1e-6)
        np.testing.assert_allclose(snr, 2 * [20], atol=1e-6)

    def test_output_sxr_more_outputs_than_sources_inf(self):
        sdr, sir, snr = output_sxr(
            morph('kKt->kKt', [
                [1 * self.s1, 0 * self.s2, 0 * self.n],
                [0 * self.s1, 1 * self.s2, 0 * self.n]
            ]),
            morph('Kt->Kt', [0 * self.n, 0 * self.n, 1 * self.n]),
            average_sources=False,
        )
        np.testing.assert_allclose(sdr, 2 * [np.inf])
        np.testing.assert_allclose(sir, 2 * [np.inf])
        np.testing.assert_allclose(snr, 2 * [np.inf])

    def test_output_sxr_more_outputs_than_sources(self):
        sdr, sir, snr = output_sxr(
            morph('kKt->kKt', [
                [10 * self.s1, 1 * self.s2, 0 * self.n],
                [0 * self.s1, 10 * self.s2, 0 * self.n]
            ]),
            morph('Kt->Kt', [10 * self.n, 0 * self.n, 0 * self.n]),
            average_sources=False,
        )
        np.testing.assert_allclose(sdr, [0, 20], atol=1e-6)
        np.testing.assert_allclose(sir, [np.inf, 20], atol=1e-6)
        np.testing.assert_allclose(snr, [0, np.inf], atol=1e-6)

    def test_output_sxr(self):
        image_contribution = morph('kKt->kKt', [
                [10 * self.s1, 1 * self.s2],
                [0 * self.s1, 10 * self.s2]
            ])
        noise_contribution = morph('Kt->Kt', [10 * self.n, 0 * self.n])
        sdr, sir, snr = output_sxr(
            image_contribution,
            noise_contribution,
            average_sources=False,
        )
        assert all(snr == output_sxr(
            image_contribution,
            noise_contribution,
            average_sources=False,
        ).snr)
        np.testing.assert_allclose(sdr, [0, 20], atol=1e-6)
        np.testing.assert_allclose(sir, [np.inf, 20], atol=1e-6)
        np.testing.assert_allclose(snr, [0, np.inf], atol=1e-6)
