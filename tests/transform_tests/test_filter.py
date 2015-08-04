import unittest
from nt.io.audioread import audioread
import numpy as np
# from scipy import signal

import nt.testing as tc

import nt.transform as transform
# from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):
    def test_offcomp(self):
        path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        y = audioread(path)
        yFilterd = transform.offcomp(y)

        tc.assert_equal(yFilterd.shape, y.shape)
        tc.assert_isreal(yFilterd)
        tc.assert_array_not_equal(yFilterd[1:], y[1:])
        tc.assert_equal(yFilterd[0], y[0])
        #ToDo: Outout predictable?

        yFilterd = transform.preemphasis(y, 0.97)

        tc.assert_equal(yFilterd.shape, y.shape)
        tc.assert_isreal(yFilterd)

        tc.assert_equal(yFilterd[0], y[0])
        #ToDo: Outout predictable?

