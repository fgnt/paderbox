import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

ger_json = db_test.ROOT / "german.json"
#ger_json = "german.json"

class TestGermanDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(ger_json)

    def test_dataset(self):
        self.assert_in_datasets(['tr_Kinect_Beam'])

    def test_len(self):
        # total length
        self.assert_total_length(71156)

        # datasets
        self.assertEqual(len(list(self.json[DATASETS])), 15,
                         "There should be 15 datasets in '{}'!"
                         .format(DATASETS))
        # dataset length
        self.assertEqual(len(list(self.json[DATASETS]['tr_Kinect_Beam'])), 11734,
                    "There should be 11734 utt_ids in dataset 'tr_Kinect_Beam'!")

    def test_examples(self):
        self.assert_in_example([TRANSCRIPTION, AUDIO_PATH])

if __name__ == '__main__':
    unittest.main()
