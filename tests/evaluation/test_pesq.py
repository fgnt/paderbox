import unittest
from nt.evaluation.pesq import pesq
import numpy
import numpy.testing as nptest
from nt.io.audioread import audioread
import os
import inspect


class TestProposedPESQ(unittest.TestCase):
    """
    This test case was written before the code was adapted.
    This is, why it fails.
    """
    def setUp(self):
        current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))

        self.ref_path = os.path.join(current_dir, 'data/speech.wav')
        self.deg_path = os.path.join(current_dir, 'data/speech_bab_0dB.wav')

        self.ref_array = audioread(self.ref_path)
        self.deg_array = audioread(self.deg_path)

    def test_wb_scores_with_lists_of_paths_length_one(self):
        scores = pesq(
            [self.ref_path],
            [self.deg_path]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))

    def test_wb_scores_with_lists_of_paths_length_two(self):
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083, 4.644]))

    def test_wb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))

    def test_wb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array]
        )
        nptest.assert_equal(scores, numpy.asarray([1.083, 4.644]))

    def test_nb_scores_with_lists_of_paths_length_one(self):
        scores = pesq(
            [self.ref_path],
            [self.deg_path],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607]))

    def test_nb_scores_with_lists_of_paths_length_two(self):
        scores = pesq(
            [self.ref_path, self.ref_path],
            [self.deg_path, self.ref_path],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607, 4.549]))

    def test_nb_scores_with_lists_of_arrays_length_one(self):
        scores = pesq(
            [self.ref_array],
            [self.deg_array],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607]))

    def test_nb_scores_with_lists_of_arrays_length_two(self):
        scores = pesq(
            [self.ref_array, self.ref_array],
            [self.deg_array, self.ref_array],
            'nb'
        )
        nptest.assert_equal(scores, numpy.asarray([1.607, 4.549]))

    def test_wb_scores_with_paths_directly(self):
        scores = pesq(
            self.ref_path,
            self.deg_path
        )
        nptest.assert_equal(scores, numpy.asarray([1.083]))
