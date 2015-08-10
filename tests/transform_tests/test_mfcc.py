import unittest
from nt.io.audioread import audioread
import numpy as np
# from scipy import signal

import nt.testing as tc

import nt.transform as transform
# from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):
    def test_mfcc(self):
        path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        y = audioread(path)
        yFilterd = transform.mfcc(y)

        tc.assert_equal(yFilterd.shape, (291, 13))
        tc.assert_isreal(yFilterd)
