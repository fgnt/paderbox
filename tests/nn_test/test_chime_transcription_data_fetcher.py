import unittest
import numpy
from nt.nn.data_fetchers.chime_transcription_data_fetcher \
    import ChimeTranscriptionDataFetcher

src = '/net/storage/2015/chime/chime_ref_data/data/json/chime.json'

class ChimeTranscriptionDataFetcherTest(unittest.TestCase):

    def setUp(self):
        self.fetcher = ChimeTranscriptionDataFetcher('chime',
                            src,
                            'train/A_database/flists/wav/channels_6/tr05_simu')

    def test_data_type(self):
        data = self.fetcher.get_data_for_indices((0,))[0]
        self.assertEqual(data.dtype, numpy.int32)