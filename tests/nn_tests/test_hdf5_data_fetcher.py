import os
import unittest

import h5py
import numpy as np

from nt.nn.data_fetchers import HDF5DataFetcher
from nt.nn import DataProvider


def _get_data(dimension):
    if dimension == 0:
        return 0
    elif dimension == 3:
        return np.arange(4 * 3 * 2).reshape(
                (4*3, 2)).reshape(3, 4, 2).transpose(1, 0, 2)
        # This has 4 time steps with three batches and the features for batch
        # 1 are [0, 1], [2, 3], [4, 5], [6, 7]
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
        self.assertIn('utt_ids',info)
        self.assertIn('frame_indices', info)

    def test_read_utterance(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'])
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 3, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))

    def test_read_utterance_context(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'])
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        for idx, feature in enumerate(data['3d']):
            ref_val = feature[2]
            # Avoid the appended zeros for the last frame
            end_idx = feature.shape[0] if idx != 3 else 3
            for val in feature[2:end_idx]:
                self.assertEqual(ref_val, val)
                ref_val += 1

    def test_read_utterance_context_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'],
                                  channels_as_batch=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        data_3d = _get_data(3)
        for idx, feature in enumerate(data['3d']):
            np.testing.assert_equal(
                    feature[2:4], data_3d[idx%4, np.floor(idx/4), :])

    def test_read_utterance_context_sequence(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'],
                                  sequence_features=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        for feature in data['3d']:
            ref_val = feature[0][2]
            for val in feature[2:]:
                self.assertEqual(ref_val, val)
                ref_val += 1

    def test_read_utterance_context_channel_as_batch_sequence(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'],
                                  channels_as_batch=True, sequence_features=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        data_3d = _get_data(3)
        for idx, feature in enumerate(data['3d']):
            np.testing.assert_equal(
                    feature[np.floor(idx/4), 2:4],
                    data_3d[idx%4, np.floor(idx/4), :])

    def test_read_utterance_context_cnn(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'],
                                  cnn_features=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        d = data['3d']
        d_ref = _get_data(3)
        for t in range(d.shape[0]):
            for c in range(d.shape[1]):
                self.assertEqual(d[t, c, 0, 1], d_ref[t, c, 0])
                self.assertEqual(d[t, c, 1, 1], d_ref[t, c, 1])

    def test_read_utterance_context_cnn_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['0d', '3d', '4d'], left_context=1,
                                  right_context=1, context_list=['3d'],
                                  cnn_features=True, channels_as_batch=True)
        data = fetcher.get_data_for_indices((0,))
        for n in ['0d', '3d', '4d']:
            self.assertIn(n, data)
        for dimension in [0, 4]:
            np.testing.assert_equal(data['{}d'.format(dimension)],
                                    _get_data(dimension))
        d = data['3d']
        d_ref = _get_data(3)
        for t in range(d.shape[0]):
            b = np.floor(t/4)
            self.assertEqual(d[t, 0, 0, 1], d_ref[t, b, 0])
            self.assertEqual(d[t, 0, 1, 1], d_ref[t, b, 1])

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
                                  ['3d'], mode='frames', cnn_features=True,
                                  context_list=['3d'])
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (1, 3, 2, 1))

    def test_read_frame_cnn_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True,
                                  cnn_features=True, context_list=['3d'])
        data = fetcher.get_data_for_indices((0,))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 1, 2, 1))

    def test_read_frames_cnn(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', cnn_features=True,
                                  context_list=['3d'])
        data = fetcher.get_data_for_indices((0, 3, 1))
        for n in ['3d']:
            self.assertIn(n, data)
        self.assertEqual(data['3d'].shape, (3, 3, 2, 1))

    def test_read_frames_cnn_channel_as_batch(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', channels_as_batch=True,
                                  cnn_features=True, context_list=['3d'])
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

    def test_integration(self):
        fetcher = HDF5DataFetcher('test', '/tmp/h5_testing_file', 'testing',
                                  ['3d'], mode='frames', left_context=3,
                                  right_context=3, context_list=['3d'],
                                  channels_as_batch=True, cnn_features=True)
        dp = DataProvider((fetcher,), batch_size=2, shuffle_data=False)
        for batch in dp.iterate():
            pass
        for batch in dp.iterate():
            break
        for batch in dp.iterate():
            pass
