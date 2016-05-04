import unittest

from nt.io.audioread import audioread
# from scipy import signal

import nt.testing as tc
import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import testing as testing_dir


class TestSTFTMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        path = testing_dir('timit', 'data', 'sample_1.wav')
        self.x = audioread(path)

    def test_offcomp(self):
        y = self.x
        yFilterd = transform.offset_compensation(y)

        tc.assert_equal(yFilterd.shape, y.shape)
        tc.assert_isreal(yFilterd)
        tc.assert_array_not_equal(yFilterd[1:], y[1:])
        tc.assert_equal(yFilterd[0], y[0])

    def test_preemphasis(self):
        y = self.x
        yFilterd = transform.preemphasis(y, 0.97)

        tc.assert_equal(yFilterd.shape, y.shape)
        tc.assert_isreal(yFilterd)
        tc.assert_equal(yFilterd[0], y[0])

    def test_preemphasis_with_offcomp(self):
        y = self.x

        y_pre = transform.preemphasis(y)
        y_ref = transform.offset_compensation(y_pre)

        y_both = transform.preemphasis_with_offset_compensation(y)

        tc.assert_almost_equal(y_ref, y_both)
