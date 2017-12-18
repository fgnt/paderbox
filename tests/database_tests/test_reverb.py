import json
import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

reverb = db_test.ROOT / "reverb.json"

class TestReverbDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(reverb)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)
        self.assertIn('RealData_dt_for_1ch_far_room1', self.json[DATASETS])

    def test_len(self):
        # ids
        self.assertEqual(len(list(self.json[EXAMPLES])), 36216,
                         "There should be 36216 utt_ids in '{}'!"
                         .format(EXAMPLES))
        # datasets
        self.assertEqual(len(list(self.json[DATASETS])), 51,
                         "There should be 52 datasets in '{}'!"
                         .format(DATASETS))
        # dataset length
        self.assertEqual(len(list(self.json[DATASETS]['RealData_dt_for_1ch_far_room1'])), 89,
                    "There should be 89 utt_ids in dataset 'RealData_dt_for_1ch_far_room1'!")

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])

if __name__ == '__main__':
    unittest.main()
