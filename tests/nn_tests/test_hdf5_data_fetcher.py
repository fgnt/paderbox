import os
import unittest

import h5py
import numpy as np

from nt.nn.data_fetchers import HDF5DataFetcher


def _get_data(dimension):
    if dimension == 0:
        return 0
    elif dimension == 3:
        return np.arange(4 * 3 * 2).reshape((4, 3, 2))
    elif dimension == 4:
        return np.arange(2 * 4 * 3 * 2).reshape((2, 4, 3, 2))


class TestH5DataFetcher(unittest.TestCase):
    def setUp(self):
        with h5py.File('/tmp/h5_testing_file', 'w') as f:
            grp = f.create_group('testing')
            grp = grp.create_group('test')
            grp.create_dataset('0d', data=_get_data(0))
            grp.create_dataset('3d', data=_get_data(3))
            grp.create_dataset('4d', data=_get_data(4))

    def tearDown(self):
        os.remove('/tmp/h5_testing_file')

    def test_len_utterance_mode(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'])
        self.assertEqual(len(fetcher), 1)

    def test_len_frame_mode(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames')
        self.assertEqual(len(fetcher), 4)

    def test_get_batch_info_utterance_mode(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'])
        info = fetcher.get_batch_info_for_indices((0,))
        self.assertIn('utt_id',info)
        self.assertEqual(info['utt_id'][0], 'test')

    def test_get_batch_info_frames_mode(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames')
        info = fetcher.get_batch_info_for_indices((0, 1))
        self.assertIn('utt_id',info)
        self.assertIn('frame_idx', info)
        for i in range(1):
            self.assertEqual(info['utt_id'][i], 'test')
            self.assertEqual(int(info['frame_idx'][i]), i)

    def test_read_utterance(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'])
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 3, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))

    def test_read_frame_value_error(self):
        with self.assertRaises(ValueError) as cm:
            HDF5DataFetcher('test', '/tmp/h5_testing_file',
                          'testing', ['0d', '3d'], mode='frames')
        self.assertEqual(
                cm.exception.args[0],
                'Frame mode is only available for data in TxMxF format but 0d has shape ()')

    def test_read_frame_ff(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames')
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (1, 2))

    def test_read_frame_ff_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3*1, 2))

    def test_read_frames_ff(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames')
        data = fetcher.get_data_for_indices((0, 3, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 2))

    def test_read_frames_ff_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True)
        data = fetcher.get_data_for_indices((0, 3, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3*3, 2))

    def test_read_frames_ff_context(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', left_context=3,
                                  right_context=3, context_list=['3d'])
        data = fetcher.get_data_for_indices((0, 2, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 7*2))

    def test_read_frames_ff_context_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', left_context=3,
                                  right_context=3, context_list=['3d'],
                                  channels_as_batch=True)
        data = fetcher.get_data_for_indices((0, 2, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3*3, 7*2))



    def test_read_frame_cnn(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', cnn_features=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (1, 3, 2, 1))

    def test_read_frame_cnn_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True,
                                  cnn_features=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 1, 2, 1))

    def test_read_frames_cnn(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', cnn_features=True)
        data = fetcher.get_data_for_indices((0, 3, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 3, 2, 1))

    def test_read_frames_cnn_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True,
                                  cnn_features=True)
        data = fetcher.get_data_for_indices((0, 3, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3*3, 1, 2, 1))

    def test_read_frames_cnn_context(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', left_context=3,
                                  right_context=3, context_list=['3d'],
                                  cnn_features=True)
        data = fetcher.get_data_for_indices((0, 2, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 3, 2, 7))

    def test_read_frames_cnn_context_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', left_context=3,
                                  right_context=3, context_list=['3d'],
                                  channels_as_batch=True, cnn_features=True)
        data = fetcher.get_data_for_indices((0, 2, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3*3, 1, 2, 7))