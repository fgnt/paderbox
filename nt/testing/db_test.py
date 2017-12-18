import unittest

from nt.io.data_dir import database_jsons as database_jsons_dir
from nt.database.keys import *

ROOT = database_jsons_dir

class DatabaseTest(unittest.TestCase):


    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])

