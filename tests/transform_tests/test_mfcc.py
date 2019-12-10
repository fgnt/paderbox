import unittest

from paderbox.io.audioread import audioread
# from scipy import signal

import paderbox.testing as tc
from paderbox.testing.testfile_fetcher import get_file_path
import paderbox.transform as transform
# from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):
    @unittest.skip("Not used at the moment, switch to e.g. librosa")
    def test_mfcc(self):
        path = get_file_path("sample.wav")

        y = audioread(path)[0]
        y_filtered = transform.mfcc(y)

        tc.assert_equal(y_filtered.shape, (291, 13))
        tc.assert_isreal(y_filtered)
