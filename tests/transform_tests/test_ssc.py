import unittest
from nt.io.audioread import audioread
import numpy as np
# from scipy import signal

import nt.testing as tc

import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import timit as timit_dir


class TestSTFTMethods(unittest.TestCase):
    def test_mfcc(self):
        path = timit_dir('pcm', 'train', 'dr1', 'fcjf0', 'sa1.wav')
        y = audioread(path)
        yFilterd = transform.ssc(y)

        tc.assert_equal(yFilterd.shape, (295, 26))
        tc.assert_isreal(yFilterd)
