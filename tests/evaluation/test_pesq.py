import unittest
from nt.evaluation.pesq import pesq, threaded_pesq
import numpy
import numpy.testing as nptest
from nt.io.audioread import audioread

class TestPESQ(unittest.TestCase):

    def setUp(self):
        self.ref = 'data/speech.wav'
        self.deg = 'data/speech_bab_0dB.wav'
        self.refer = audioread(self.ref)
        self.rate = 16000

    def test_wb_scores(self):
        scores = pesq(self.ref, self.deg, 'wb', self.rate)
        nptest.assert_equal(scores, numpy.asarray([[0, 1.083]]))

    def test_nb_scores(self):
        scores = pesq(self.ref, self.deg, 'nb', self.rate)
        nptest.assert_equal(scores, numpy.asarray([[1.969, 1.607]]))

    def test_wb_scores_thread(self):
        scores = threaded_pesq(3*[self.ref], 3*[self.deg], 'wb', self.rate)
        nptest.assert_equal(scores, numpy.asarray(3*[[0, 1.083]]))

    def test_nb_scores_thread(self):
        scores = threaded_pesq(3*[self.ref], 3*[self.deg], 'nb', self.rate)
        nptest.assert_equal(scores, numpy.asarray(3*[[1.969, 1.607]]))