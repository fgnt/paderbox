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
            length=10, dimension=2
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
        assert len(df) == 10

    def test_frame_mode_length(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        assert len(df) == 100

    def test_frame_mode_iterate(self):
        df = AlmostIdentityCallbackFetcher('identity', mode='frames')
        dp = data_provider.DataProvider((df,), 10)
        for _ in dp.iterate(fork_fetchers=False):
            pass
        dp = data_provider.DataProvider((df,), 7)
        for _ in dp.iterate(fork_fetchers=False):
            pass
