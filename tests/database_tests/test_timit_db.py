import json
import unittest

from nt.io import load_json
from nt.testing import db_test
from nt.database.keys import *

timit_json = db_test.ROOT / "timit.json"

class TestTimitDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(timit_json)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])

    def test_len(self):
        self.assertEqual(len(self.json[DATASETS]['test']), 1680,
                         "There should be 1680 files in 'test'!")
        self.assertEqual(len(self.json[DATASETS]['train']), 4620,
                         "There should be 4620 files in 'train'!")
        self.assertEqual(len(self.json[DATASETS]['test_core']),192,
                         "There should be 192 files in 'test_cor'!")

    # def test_orth_len(self):
    #     phoneme_len = len(self.json['orth']['phoneme'])
    #     word_len = len(self.json['orth']['word'])
    #     self.assertEqual(phoneme_len, 6300, "Databank has wrong number of phnome transciprions!")
    #     self.assertEqual(word_len, 6300, "Databank has wrong number of word transciprions!")

if __name__ == '__main__':
    unittest.main()
