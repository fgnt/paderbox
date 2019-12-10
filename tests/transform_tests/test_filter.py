import unittest

from paderbox.io.audioread import audioread
# from scipy import signal

import paderbox.testing as tc
from paderbox.testing.testfile_fetcher import get_file_path
import paderbox.transform as transform
# from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        path = get_file_path("sample.wav")

        self.x = audioread(path)[0]

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
