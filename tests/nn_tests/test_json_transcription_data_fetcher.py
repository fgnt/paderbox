import unittest
import numpy
from nt.nn.data_fetchers.json_transcription_data_fetcher \
    import JsonTranscriptionDataFetcher
from nt.transcription_handling.transcription_handler import write_lexicon
import json

from nt.io.data_dir import database_jsons as database_jsons_dir

JSON_SRC = database_jsons_dir.join('chime.json')

class JsonTranscriptionDataFetcherTest(unittest.TestCase):

    def setUp(self):
        with open(JSON_SRC) as fid:
            src = json.load(fid)
        lexicon = write_lexicon(src)
        self.fetcher = JsonTranscriptionDataFetcher(
            'chime', src, 'train/A_database/flists/wav/channels_6/tr05_simu',
            lexicon)

    def test_data_type(self):
        data = self.fetcher.get_data_for_indices((0,))['chime']
        self.assertEqual(data.dtype, numpy.int32)