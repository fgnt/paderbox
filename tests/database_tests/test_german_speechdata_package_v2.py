import json
import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

ger_json = db_test.ROOT / "german.json"

class test_ger_db(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(ger_json)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)
        self.assertIn('tr_Kinect_Beam', self.json[DATASETS])

    def test_len(self):
        # ids
        self.assertEqual(len(list(self.json[EXAMPLES])), 71156,
                         "There should be 71156 utt_ids in '{}'!"
                         .format(EXAMPLES))
        # datasets
        self.assertEqual(len(list(self.json[DATASETS])), 15,
                         "There should be 15 datasets in '{}'!"
                         .format(DATASETS))
        # dataset length
        self.assertEqual(len(list(self.json[DATASETS]['tr_Kinect_Beam'])), 11734,
                    "There should be 11734 utt_ids in dataset 'tr_Kinect_Beam'!")

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])

if __name__ == '__main__':
    unittest.main()
