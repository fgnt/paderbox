import os
import unittest

import h5py
import numpy as np
from nt.dataset.hdf5_dataset import UtteranceHDF5Dataset


def _get_data(dimension, deltas=0):
    if dimension == 0:
        data = 0
    elif dimension == 2:
        data = np.arange(4 * 2).reshape(4, 2)
        # This has 4 time steps where the features are
        # [0, 1], [2, 3], [4, 5], [6, 7]
    elif dimension == 3:
        # TODO: Why two times reshape?
        data = np.arange(4 * 3 * 2).reshape(
                (4*3, 2)).reshape(3, 4, 2).transpose(1, 0, 2)
        # This has 4 time steps with three batches and the features for batch
        # 1 are [0, 1], [2, 3], [4, 5], [6, 7]
    elif dimension == 4:
        data = np.arange(2 * 4 * 3 * 2).reshape((2, 4, 3, 2))
    else:
        raise NotImplementedError

    if deltas > 0:
        data = np.concatenate(deltas*[data], axis=-1)
    return data


class TestUtteranceHDF5Dataset(unittest.TestCase):
    def setUp(self):
        with h5py.File('/tmp/h5_testing_file.h5', 'w') as f:
            grp = f.create_group('testing')
            grp = grp.create_group('test')
            grp.create_dataset('0d', data=_get_data(0))
            grp.create_dataset('2d', data=_get_data(2))
            grp.create_dataset('2d_delta', data=_get_data(2, 1))
            grp.create_dataset('2d_delta_delta', data=_get_data(2, 2))
            grp.create_dataset('3d', data=_get_data(3))
            grp.create_dataset('4d', data=_get_data(4))

        self.dataset = UtteranceHDF5Dataset(
            hdf5_file='/tmp/h5_testing_file.h5',
            path='testing',
            data_list=['0d', '3d', '4d']
        )

    def tearDown(self):
        os.remove('/tmp/h5_testing_file.h5')

    def test_len_utterance_mode(self):
        self.assertEqual(len(self.dataset), 1)

    def test_read_utterance(self):
        data = self.dataset.get_example(0)
        for key, dimension in zip(['0d', '3d', '4d'], [0, 3, 4]):
            self.assertIn(key, data)
            np.testing.assert_equal(
                data[key], _get_data(dimension)
            )
