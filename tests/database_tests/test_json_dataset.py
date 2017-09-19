import unittest

import numpy as np
from nt.dataset.json_dataset import ContextUtteranceJsonCallbackDataset
from nt.dataset.json_dataset import UtteranceJsonCallbackDataset

from nt.testing import db_test

_chime_json = db_test.ROOT/ 'chime.json'
flist = 'train/flists/wav/tr05_real'
feature_channels = ['embedded/CH1']
annotations = 'train/annotations/tr05_real'
start = 'start'
end = 'end'
context_length = 5
sample_rate = 16000


class TestContextUtteranceJsonCallbackDataset(unittest.TestCase):
    def setUp(self):

        self.dataset_with_context = ContextUtteranceJsonCallbackDataset(
            json_src=_chime_json,
            flist=flist,
            feature_channels=feature_channels,
            annotations=annotations,
            audio_start_key=start,
            audio_end_key=end,
            context_length=context_length,
            sample_rate=sample_rate
        )

        self.dataset = UtteranceJsonCallbackDataset(
            json_src=_chime_json,
            flist=flist,
            feature_channels=['observed/CH1']
        )

    def test_len_utterance_mode(self):
        self.assertEqual(len(self.dataset_with_context), 1600)
        self.assertEqual(len(self.dataset_with_context), len(self.dataset))

    def test_read_utterance(self):
        data_with_context = self.dataset_with_context.get_example(0)
        data = self.dataset.get_example(0)
        context = data_with_context['_context_samples'].sum(dtype=np.int)
        assert context == sample_rate * context_length, \
            'Actual context {} does not match expected context {}'.format(context, sample_rate * context_length)
        np.testing.assert_equal(data['observed'].shape[-1],
                                data_with_context['embedded'].shape[-1] - context)
        np.testing.assert_equal(data['observed'], data_with_context['embedded'][:, context:])
