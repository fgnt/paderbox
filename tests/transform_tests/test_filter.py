import unittest
from nt.io.audioread import audioread
import numpy as np
# from scipy import signal

import nt.testing as tc

import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import timit as timit_dir

class TestSTFTMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        path = timit_dir('pcm', 'train', 'dr1', 'fcjf0', 'sa1.wav')
        self.x = audioread(path)

    def test_offcomp(self):
        y = self.x
        yFilterd = transform.offcomp(y)

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
