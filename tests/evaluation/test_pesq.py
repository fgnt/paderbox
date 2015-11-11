import unittest
from nt.evaluation.pesq import pesq, threaded_pesq
import numpy
import numpy.testing as nptest
from nt.io.audioread import audioread
import os
import inspect


class TestPESQ(unittest.TestCase):

    def setUp(self):
        current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))

        self.ref = os.path.join(current_dir, 'data/speech.wav')
        self.deg = os.path.join(current_dir, 'data/speech_bab_0dB.wav')
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
