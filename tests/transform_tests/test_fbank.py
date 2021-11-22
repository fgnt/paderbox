import unittest

import numpy as np

from paderbox.io.audioread import audioread
# from scipy import signal

import paderbox.testing as tc
from paderbox.testing.testfile_fetcher import get_file_path
import paderbox.transform as transform
# from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):

    def test_fbank(self):
        path = get_file_path("sample.wav")

        y = audioread(path)[0]
        feature = transform.fbank(y)

        tc.assert_equal(feature.shape, (240, 23))
        tc.assert_isreal(feature)
        tc.assert_array_greater_equal(feature, 0)

    def test_hz2mel_htk(self):
        tc.assert_equal(transform.module_fbank.hz2mel(6300), 2595)
        tc.assert_equal(transform.module_fbank.hz2mel(
            np.array([6300, 6300, 6300])), 2595)

    def test_mel2hz_htk(self):
        tc.assert_equal(transform.module_fbank.mel2hz(2595), 6300)
        tc.assert_equal(transform.module_fbank.mel2hz(
            np.array([2595, 2595, 2595])), 6300)

    def test_hz2mel2hz_htk(self):
        hz = np.random.rand(5, 5) * 4000
        mels = transform.module_fbank.hz2mel(hz, htk_mel=True)
        tc.assert_almost_equal(
            hz, transform.module_fbank.mel2hz(mels, htk_mel=True),
        )

    def test_mel2hz2mel_htk(self):
        mels = np.random.rand(5, 5) * 4000
        hz = transform.module_fbank.mel2hz(mels, htk_mel=True,)
        tc.assert_almost_equal(
            mels, transform.module_fbank.hz2mel(hz, htk_mel=True),
        )

    def test_hz2mel2hz_slaney(self):
        hz = np.random.rand(5, 5) * 4000
        mels = transform.module_fbank.hz2mel(hz, htk_mel=False)
        tc.assert_almost_equal(
            hz, transform.module_fbank.mel2hz(mels, htk_mel=False),
        )

    def test_mel2hz2mel_slaney(self):
        mels = np.random.rand(5, 5) * 4000
        hz = transform.module_fbank.mel2hz(mels, htk_mel=False,)
        tc.assert_almost_equal(
            mels, transform.module_fbank.hz2mel(hz, htk_mel=False),
        )
