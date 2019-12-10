import unittest

from paderbox.io.audioread import audioread
# from scipy import signal

import paderbox.testing as tc
import paderbox.transform as transform
# from pymatbridge import Matlab



class TestSTFTMethods(unittest.TestCase):
    @unittest.skip("Not used at the moment, switch to e.g. librosa")
    def test_mfcc(self):
        path = tc.fetch_file_from_url(
            "https://github.com/fgnt/pb_test_data/blob/master"
            "/bss_data/low_reverberation/speech_source_0.wav",
            "speech_source_0.wav"
        )
        y = audioread(path)[0]
        y_filtered = transform.mfcc(y)

        tc.assert_equal(y_filtered.shape, (291, 13))
        tc.assert_isreal(y_filtered)
