import unittest
import numpy as np
import paderbox.testing as tc
from paderbox.math.vector import cos_similarity


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity(self):
        phi = 60 / 360 * 2 * np.pi

        W1 = np.array([4, 0, 0])
        W2 = np.array([1, 0, 0])
        W3 = np.array([0, 1, 0])
        W4 = np.array([1, 1j, 0])
        W5 = np.array([1, -1j, 0])
        W6 = np.array([np.cos(phi), np.sin(phi), 0])

        tc.assert_allclose(cos_similarity(W1, W1), 1.0, rtol=1e-15, atol=1e-15)
        tc.assert_allclose(cos_similarity(W1, W2), 1.0, rtol=1e-15, atol=1e-15)
        tc.assert_allclose(cos_similarity(W1, W3), 0.0, rtol=1e-15, atol=1e-15)
        tc.assert_allclose(cos_similarity(W4, W5), 0.0, rtol=1e-15, atol=1e-15)
        tc.assert_allclose(cos_similarity(W1, W6), np.cos(phi), rtol=1e-15, atol=1e-15)
