import sys
import pytest
import unittest
import os
import time
import paderbox as pb
from paderbox.io.audioread import audioread
from paderbox.io.audiowrite import audiowrite

import numpy
import numpy.testing as nptest

signal = numpy.random.uniform(-1, 1, size=(10000,))
path = 'audiowrite_test.wav'

int16_max = numpy.iinfo(numpy.int16).max

@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="`pb.io.audioread.audioread` is deprecated and does not work on"
           "windows, because wavefile needs `libsndfile-1.dll`."
           "Use `pb.io.load_audio` on windows.")
class AudioWriteTest(unittest.TestCase):

    def test_write_read_float(self):
        pb.io.dump_audio(signal, path, normalize=False)
        read_data = pb.io.load_audio(path)
        # Quantization: By default audio is saved with 16 Bit, hence an error
        # of 2**-15 is ok.
        nptest.assert_allclose(signal, read_data, atol=2**-15, rtol=0)

        audiowrite(signal, path, threaded=False)
        read_data = audioread(path)[0]
        # 0.01 is bad, but audiowrite is deprecated, load and dum audio doesn't have this issue
        nptest.assert_almost_equal(signal, read_data, decimal=2)


    def test_write_read_int(self):
        audiowrite((signal*int16_max).astype(numpy.int), path, threaded=False)
        read_data = audioread(path)[0]
        nptest.assert_almost_equal(signal, read_data, decimal=3)

    def test_write_read_complex(self):
        with nptest.assert_raises(AssertionError):
            audiowrite((signal*int16_max).astype(numpy.complex128), path, threaded=False)

    def test_write_threaded(self):
        audiowrite(signal, path)
        time.sleep(5)
        self.assertTrue(os.path.exists(path))

    def test_clipping(self):
        self.assertGreater(audiowrite(signal*1.1, path), 1)

    def tearDown(self):
        try:
            os.remove(path)
        except Exception:
            pass