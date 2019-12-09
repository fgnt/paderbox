import unittest
import numpy as np
import paderbox.testing as tc
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix


def rand(*shape, data_type=np.float64):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return uniform(data_type)
    elif data_type is np.complex64:
        return uniform(np.float32) + 1j * uniform(np.float32)
    elif data_type is np.complex128:
        return uniform(np.float64) + 1j * uniform(np.float64)


class TestCovariance(unittest.TestCase):
    def generate_data(self, x_shape, mask_shape):
        x = rand(x_shape, data_type=np.complex128)
        mask = np.random.uniform(0, 1, mask_shape)
        mask = mask / np.sum(mask, axis=0, keepdims=True)
        return x, mask

    def test_covariance_without_mask(self):
        x = rand(3, 4)
        psd = get_power_spectral_density_matrix(x)
        tc.assert_equal(psd.shape, (3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask(self):
        x = rand(3, 4)
        mask = np.random.uniform(0, 1, (4,))
        psd = get_power_spectral_density_matrix(x, mask)
        tc.assert_equal(psd.shape, (3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask_with_source(self):
        x = rand(3, 4)
        mask = np.random.uniform(0, 1, (2, 4))
        psd = get_power_spectral_density_matrix(x[None, ...], mask)
        tc.assert_equal(psd.shape, (2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_with_mask_independent_dim(self):
        x = rand(2, 3, 4)
        mask = np.random.uniform(0, 1, (2, 4,))
        psd = get_power_spectral_density_matrix(x, mask)
        tc.assert_equal(psd.shape, (2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_covariance_without_mask_independent_dim(self):
        x = rand(1, 2, 3, 4)
        psd = get_power_spectral_density_matrix(x)
        tc.assert_equal(psd.shape, (1, 2, 3, 3))
        tc.assert_positive_semidefinite(psd)

    def test_multiple_sources_for_source_separation(self):
        x = rand(2, 3, 4)
        mask = np.random.uniform(0, 1, (5, 2, 4,))
        psd = get_power_spectral_density_matrix(x[np.newaxis, ...], mask)
        tc.assert_equal(psd.shape, (5, 2, 3, 3))
        tc.assert_positive_semidefinite(psd)
