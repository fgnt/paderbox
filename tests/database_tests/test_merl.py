import json
import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

merl = db_test.ROOT / "merl_speaker_mixtures_min_8k.json"

class test_ger_db(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(merl)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)
        self.assertIn('mix_2_spk_min_cv', self.json[DATASETS])

    def test_len(self):
        # ids
        self.assertEqual(len(list(self.json[EXAMPLES])), 56000,
                         "There should be 56000 utt_ids in '{}'!"
                         .format(EXAMPLES))
        # datasets
        self.assertEqual(len(list(self.json[DATASETS])), 6,
                         "There should be 6 datasets in '{}'!"
                         .format(DATASETS))
        # dataset length
        self.assertEqual(len(list(self.json[DATASETS]['mix_2_spk_min_cv'])), 5000,
                    "There should be 5000 utt_ids in dataset 'mix_2_spk_min_cv'!")

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])

if __name__ == '__main__':
    unittest.main()
