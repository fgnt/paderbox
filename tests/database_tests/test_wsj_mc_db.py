import unittest

from nt.io import load_json
from nt.testing import db_test
from nt.database.keys import *

wsj_mc_json = db_test.ROOT / "wsj_mc.json"


class TestWSJMCDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(wsj_mc_json)

    def test_examples(self):
        self.assert_in_example([AUDIO_PATH, TRANSCRIPTION, SPEAKER_ID])

    def test_len(self):
        self.assert_total_length(2508)

        self.assertEqual(len(self.json[DATASETS]['olap_dev_5k']), 178,
                         f'Dataset olap_dev_5k should contain 178 examples')
        self.assertEqual(len(self.json[DATASETS]['move_dev_a']), 66,
                         f'Dataset move_dev_a should contain 66 examples')
        self.assertEqual(len(self.json[DATASETS]['move_dev_5k']), 148,
                         f'Dataset move_dev_5k should contain 148 examples')
        self.assertEqual(len(self.json[DATASETS]['move_dev_20k']), 156,
                         f'Dataset move_dev_20k should contain 156 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_dev_a']), 82,
                         f'Dataset stat_dev_a should contain 82 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_dev_5k']), 179,
                         f'Dataset stat_dev_5k should contain 179 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_dev_20k']), 184,
                         f'Dataset stat_dev_20k should contain 184 examples')
        self.assertEqual(len(self.json[DATASETS]['olap_ev1_5k']), 142,
                         f'Dataset olap_ev1_5k should contain 142 examples')
        self.assertEqual(len(self.json[DATASETS]['move_ev1_a']), 85,
                         f'Dataset move_ev1_a should contain 85 examples')
        self.assertEqual(len(self.json[DATASETS]['move_ev1_5k']), 190,
                         f'Dataset move_ev1_5k should contain 190 examples')
        self.assertEqual(len(self.json[DATASETS]['move_ev1_20k']), 198,
                         f'Dataset move_ev1_20k should contain 198 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev1_a']), 81,
                         f'Dataset stat_ev1_a should contain 81 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev1_5k']), 188,
                         f'Dataset stat_ev1_5k should contain 188 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev1_20k']), 193,
                         f'Dataset stat_ev1_20k should contain 193 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev2_a']), 80,
                         f'Dataset stat_ev2_a should contain 80 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev2_5k']), 183,
                         f'Dataset stat_ev2_5k should contain 183 examples')
        self.assertEqual(len(self.json[DATASETS]['stat_ev2_20k']), 175,
                         f'Dataset stat_ev2_20k should contain 175 examples')

if __name__ == '__main__':
    unittest.main()
