import unittest

from nt.io.audioread import audioread
# from scipy import signal

import nt.testing as tc
import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import testing as testing_dir


class TestSTFTMethods(unittest.TestCase):
    @unittest.skip("Not used at the moment, switch to e.g. librosa")
    def test_mfcc(self):
        path = testing_dir / 'timit' / 'data' / 'sample_1.wav'
        y = audioread(path)[0]
        yFilterd = transform.ssc(y)

        tc.assert_equal(yFilterd.shape, (294, 26))
        tc.assert_isreal(yFilterd)
