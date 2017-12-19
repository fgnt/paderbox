import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

reverb = db_test.ROOT / "reverb.json"


class TestReverbDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(reverb)


    def test_dataset(self):
        self.assert_in_datasets('RealData_dt_for_1ch_far_room1')

    def test_len(self):
        # datasets
        self.assertEqual(len(list(self.json[DATASETS])), 51,
                         "There should be 52 datasets in '{}'!"
                         .format(DATASETS))
        # dataset length
        self.assertEqual(len(list(self.json[DATASETS]['RealData_dt_for_1ch_far_room1'])), 89,
                    "There should be 89 utt_ids in dataset 'RealData_dt_for_1ch_far_room1'!")

    def test_examples(self):
        self.assert_in_example([TRANSCRIPTION, AUDIO_PATH])

if __name__ == '__main__':
    unittest.main()
