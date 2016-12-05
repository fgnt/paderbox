import unittest

import numpy as np

from nt.io.audioread import audioread
# from scipy import signal

import nt.testing as tc
import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import testing as testing_dir


class TestSTFTMethods(unittest.TestCase):

    def test_fbank(self):
        path = testing_dir / 'timit' / 'data' / 'sample_1.wav'
        y = audioread(path)
        feature = transform.fbank(y)

        tc.assert_equal(feature.shape, (291, 23))
        tc.assert_isreal(feature)
        tc.assert_array_greater_equal(feature, 0)

    def test_get_filterbanks(self):
        fbank = transform.module_fbank.get_filterbanks()

        tc.assert_equal(fbank.shape, (20, 513))
        tc.assert_isreal(fbank)
        tc.assert_array_greater_equal(fbank, 0)
        tc.assert_array_less_equal(fbank, 1)

    def test_hz2mel(self):
        tc.assert_equal(transform.module_fbank.hz2mel(6300), 2595)
        tc.assert_equal(transform.module_fbank.hz2mel(
            np.array([6300, 6300, 6300])), 2595)

    def test_mel2hz(self):
        tc.assert_equal(transform.module_fbank.mel2hz(2595), 6300)
        tc.assert_equal(transform.module_fbank.mel2hz(
            np.array([2595, 2595, 2595])), 6300)

    def test_mel2hzandhz2mel(self):
        rand = np.random.rand(5, 5) * 1000
        tc.assert_almost_equal(
            rand, transform.module_fbank.mel2hz(
                transform.module_fbank.hz2mel(rand)))
        tc.assert_almost_equal(rand, transform.module_fbank.hz2mel(transform.module_fbank.mel2hz(rand)))
