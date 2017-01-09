import unittest

from nt.io.audioread import audioread
# from scipy import signal

import nt.testing as tc
import nt.transform as transform
# from pymatbridge import Matlab

from nt.io.data_dir import testing as testing_dir


class TestSTFTMethods(unittest.TestCase):

    def test_mfcc(self):
        path = testing_dir / 'timit' / 'data' / 'sample_1.wav'
        y = audioread(path)
        y_filtered = transform.mfcc(y)

        tc.assert_equal(y_filtered.shape, (291, 13))
        tc.assert_isreal(y_filtered)
