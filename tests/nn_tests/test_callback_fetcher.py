from nt.nn.data_fetchers.callback_fetcher import CallbackDataFetcher
import numpy as np
import numpy.testing
import unittest
from nt.nn import data_provider


class AlmostIdentityCallbackFetcher(CallbackDataFetcher):
    def __init__(
            self, name, mode='utterance',
            add_context_to=None,
            sequence_features=False,
            left_context=0, right_context=0, step=1,
            cnn_features=False, deltas_as_channel=False,
            num_deltas=2,
            shuffle_frames=True, buffer_size=100,
            transformation_callback=None,
            transformation_kwargs=None, verbose=True,
            length=8, dimension=2
    ):
        self.length = length
        self.dimension = dimension
        super().__init__(
            name, mode=mode,
            add_context_to=add_context_to,
            sequence_features=sequence_features,
            left_context=left_context, right_context=right_context, step=step,
            cnn_features=cnn_features, deltas_as_channel=deltas_as_channel,
            num_deltas=num_deltas,
            shuffle_frames=shuffle_frames, buffer_size=buffer_size,
            transformation_callback=transformation_callback,
            transformation_kwargs=transformation_kwargs, verbose=verbose
        )

    def _get_utterance_list(self):
        """ Returns a list with utterance ids

        :return: List with utterance ids
        """
        return list(range(self.length))

    def _read_utterance(self, utt):
        """ Reads the (untransformed) data for utterance `utt`

        :param utt:
        :return: dict with data
        """
        X = np.arange(0.0, 1.0, 0.1) + utt
        X = np.reshape(X, (-1,) + (self.dimension-1) * (1,))
        return {'X': X}


class TestCallbackFetcher(unittest.TestCase):
    def test_utterance_mode_length(self):
        df = AlmostIdentityCallbackFetcher('identity')
        assert len(df) == 8

    def test_frame_mode_length(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        assert len(df) == 80

    def test_frame_mode_iterate(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        dp = data_provider.DataProvider((df,), 10)
        for batch in dp.iterate(fork_fetchers=False):
            assert 'X' in batch
        dp = data_provider.DataProvider((df,), 7)
        for batch in dp.iterate(fork_fetchers=False):
            assert 'X' in batch

    def test_frame_mode_iterate_fork(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        dp = data_provider.DataProvider((df,), 10)
        for batch in dp.iterate(fork_fetchers=True):
            assert 'X' in batch
        dp = data_provider.DataProvider((df,), 7)
        for batch in dp.iterate(fork_fetchers=True):
            if not 'X' in batch:
                print(batch)
            assert 'X' in batch

    def test_utterance_mode_iterate(self):
        df = AlmostIdentityCallbackFetcher('identity')
        dp = data_provider.DataProvider((df,), 1)
        for batch in dp.iterate(fork_fetchers=False):
            self.assertIn('X', batch)

    def test_utterance_mode_iterate_fork(self):
        df = AlmostIdentityCallbackFetcher('identity')
        dp = data_provider.DataProvider((df,), 1)
        for batch in dp.iterate(fork_fetchers=True):
            self.assertIn('X', batch)

    # Tests format of info and correctness of at least the first item returned
    # by _get_utterance_list. A test like this should be done for every subclass
    # of CallbackFetcher.
    def test_utterance_mode_get_batch_info(self):
        df = AlmostIdentityCallbackFetcher('identity')
        info = df.get_batch_info_for_indices((0,))
        self.assertIn('utt_id', info)
        self.assertEqual(info['utt_id'][0], 0)

    # can be moved from test_hdf5_data_fetcher; doesn't test content of info
    # but correct format provided by CallbackFetcher
    def test_frame_mode_get_batch_info(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        info=df.get_batch_info_for_indices((0,))
        self.assertIn('utt_ids', info)
        self.assertIn('frame_indices', info)

    def test_utterance_mode_get_data_for_indices_callback(self):
        df = AlmostIdentityCallbackFetcher(
            'identity',
            length=10,
            transformation_callback=lambda t, **kwargs: {'X': t['X'] + 1}
        )
        data = df.get_data_for_indices((0,))
        np.testing.assert_almost_equal(
            data['X'].flatten(),
            np.arange(0, 10) / 10 + 1
        )

    def test_utterance_mode_get_data_context(self):
        df = AlmostIdentityCallbackFetcher(
            'identity',
            left_context=1,
            right_context=1,
            add_context_to=['X']
        )
        data = df.get_data_for_indices((0,))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (10, 3))

    def test_frame_mode_get_data_context(self):
        df = AlmostIdentityCallbackFetcher(
            'identity',
            mode='frames',
            left_context=1,
            right_context=1,
            add_context_to=['X']
        )
        data = df.get_data_for_indices((0, 1))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (2, 3))

    def test_frame_mode_add_context_value_error(self):
        with self.assertRaises(AssertionError) as cm:
            AlmostIdentityCallbackFetcher(
                'identity',
                dimension=1,
                add_context_to=['X'],
                mode='frames'
            )

        self.assertEqual(
            cm.exception.args[0],
            'Only 2d arrays in the TxF format can be used to add context '
                'in frames mode.')

    def test_frame_mode_read_cnn(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                           cnn_features=True,
                                           add_context_to=['X'],
                                           deltas_as_channel=False)
        data = df.get_data_for_indices((0,))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (1, 1, 1, 1))

    def test_frame_mode_read_cnn_multiple(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                           cnn_features=True,
                                           add_context_to=['X'],
                                           deltas_as_channel=False)
        data = df.get_data_for_indices((0, 1, 2))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (3, 1, 1, 1))

    def test_frame_mode_cnn_channel_delta(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                           deltas_as_channel=True,
                                           add_context_to=['X'],
                                           cnn_features=True, num_deltas=1)
        data = df.get_data_for_indices((0, 3, 1))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (3, 2, 0, 1))

    def test_frame_mode_cnn_context(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                    left_context=3,
                                    right_context=3, add_context_to=['X'],
                                    cnn_features=True)
        data = df.get_data_for_indices((0, 2, 1))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (3, 1, 1, 7))

    def test_frame_mode_cnn_big_context(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                    left_context=10,
                                    right_context=10, add_context_to=['X'],
                                    cnn_features=True)
        data = df.get_data_for_indices((0, 2, 1))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (3, 1, 1, 21))

    def test_frame_mode_cnn_context_delta(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames',
                                    num_deltas=1, left_context=3,
                                    deltas_as_channel=True,
                                    right_context=3, add_context_to=['X'],
                                    cnn_features=True)
        data = df.get_data_for_indices((3, 2, 1))
        self.assertIn('X', data)
        self.assertEqual(data['X'].shape, (3, 2, 0, 7))

    def test_frame_buffer_stays_empty_even_after_reset(self):
        """
        Callback fetcher used to load the next data matrix in utterance mode.
        This happened in reset but is only needed for frames mode.
        """
        df = AlmostIdentityCallbackFetcher('identity')
        _ = df.get_data_for_indices((0,))

        with self.assertRaises(AttributeError):
            print(df.frame_buffer)

        df.reset()
        with self.assertRaises(AttributeError):
            print(df.frame_buffer)
