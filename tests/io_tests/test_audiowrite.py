import numpy
import numpy.testing as nptest
import unittest
import os
import time
from nt.io.audioread import audioread
from nt.io.audiowrite import audiowrite

signal = numpy.random.uniform(-1, 1, size=(10000,))
path = 'audiowrite_test.wav'

int16_max = numpy.iinfo(numpy.int16).max


class AudioWriteTest(unittest.TestCase):

    def test_write_read_float(self):
        audiowrite(signal, path, threaded=False)
        read_data = audioread(path)
        nptest.assert_almost_equal(signal, read_data, decimal=3)

    def test_write_read_int(self):
        audiowrite((signal*int16_max).astype(numpy.int), path, threaded=False)
        read_data = audioread(path)
        nptest.assert_almost_equal(signal, read_data, decimal=5)

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