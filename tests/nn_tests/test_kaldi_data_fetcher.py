import unittest
import numpy as np
from nt.nn.data_fetchers import KaldiDataFetcher

SCP = '/net/storage/python_unittest_data/kaldi_data_fetcher/feats.scp'
ALI = '/net/storage/python_unittest_data/kaldi_data_fetcher/'
MODEL = '/net/storage/python_unittest_data/kaldi_data_fetcher/final.mdl'


class TestKaldiDataFetcher(unittest.TestCase):

    def test_utterance_mode(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].shape[1], 13)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])

    def test_utterance_mode_context(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL, left_context=2,
                                   right_context=3)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].shape[1], 6*13)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])

    def test_utterance_mode_sequential(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL,
                                   sequence_features=True)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].ndim, 3)
        self.assertEqual(data['x'].shape[2], 13)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])

    def test_utterance_mode_sequential_context(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL,
                                   sequence_features=True, left_context=2,
                                   right_context=3)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].ndim, 3)
        self.assertEqual(data['x'].shape[2], 6*13)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])

    def test_frame_mode(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL, mode='frames')
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].shape[1], 13)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])
        self.assertEqual(data['x'].ndim, 2)
        self.assertEqual(data['ali'].ndim, 2)

    def test_frame_mode_context(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL, mode='frames',
                                   left_context=2, right_context=3)
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].shape[1], 6*13)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])
        self.assertEqual(data['x'].ndim, 2)
        self.assertEqual(data['ali'].ndim, 2)

    def test_frame_mode_context_cnn_features(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, ALI, MODEL, mode='frames',
                                   left_context=2, right_context=3,
                                   cnn_features=True)
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertIn('ali', data)
        self.assertEqual(data['x'].shape[1], 1)
        self.assertEqual(data['x'].shape[2], 13)
        self.assertEqual(data['x'].shape[3], 6)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].shape[0], data['ali'].shape[0])
        self.assertEqual(data['x'].ndim, 4)
        self.assertEqual(data['ali'].ndim, 2)

    def test_utterance_mode_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertEqual(data['x'].shape[1], 13)

    def test_utterance_mode_context_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, left_context=2,
                                   right_context=3)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertEqual(data['x'].shape[1], 6*13)

    def test_utterance_mode_sequential_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP,
                                   sequence_features=True)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertEqual(data['x'].ndim, 3)
        self.assertEqual(data['x'].shape[2], 13)

    def test_utterance_mode_sequential_context_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP,
                                   sequence_features=True, left_context=2,
                                   right_context=3)
        data = fetcher.get_data_for_indices((0,))
        self.assertIn('x', data)
        self.assertEqual(data['x'].ndim, 3)
        self.assertEqual(data['x'].shape[2], 6*13)

    def test_frame_mode_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, mode='frames')
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertEqual(data['x'].shape[1], 13)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].ndim, 2)

    def test_frame_mode_context_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, mode='frames',
                                   left_context=2, right_context=3)
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertEqual(data['x'].shape[1], 6*13)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].ndim, 2)

    def test_frame_mode_context_cnn_features_no_ali(self):
        fetcher = KaldiDataFetcher('kaldi', SCP, mode='frames',
                                   left_context=2, right_context=3,
                                   cnn_features=True)
        data = fetcher.get_data_for_indices((0, 3, 10))
        self.assertIn('x', data)
        self.assertEqual(data['x'].shape[1], 1)
        self.assertEqual(data['x'].shape[2], 13)
        self.assertEqual(data['x'].shape[3], 6)
        self.assertEqual(data['x'].shape[0], 3)
        self.assertEqual(data['x'].ndim, 4)