import unittest
import numpy
from nt.nn.data_fetchers.json_transcription_data_fetcher \
    import JsonTranscriptionDataFetcher
import json

JSON_SRC = '/net/storage/2015/chime/chime_ref_data/data/json/chime.json'

class JsonTranscriptionDataFetcherTest(unittest.TestCase):

    def setUp(self):
        with open(JSON_SRC) as fid:
            src = json.load(fid)
        self.fetcher = JsonTranscriptionDataFetcher('chime',
                                                    src,
                            'train/A_database/flists/wav/channels_6/tr05_simu')

    def test_data_type(self):
        data = self.fetcher.get_data_for_indices((0,))[0]
        self.assertEqual(data.dtype, numpy.int32)