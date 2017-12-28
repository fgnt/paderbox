import unittest

from nt.database.keys import *
from nt.io import load_json
from nt.testing import db_test

chime_json = db_test.ROOT / "chime.json"


class TestChimeDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(chime_json)

    def test_examples(self):
        self.assert_in_example([AUDIO_PATH, TRANSCRIPTION, START, END])

    def test_dataset(self):
        self.assert_in_datasets(['tr05_org', 'tr05_simu', 'tr05_real',
                                 'dt05_simu', 'dt05_real', 'et05_simu',
                                 'et05_real'])

    def test_len(self):
        self.assert_total_length(21796)

        self.assertEqual(len(self.json[DATASETS]['tr05_org']), 7138,
                         f'Dataset tr05_org should contain 7138 examples')
        self.assertEqual(len(self.json[DATASETS]['tr05_simu']), 7138,
                         f'Dataset tr05_simu should contain 7138 examples')
        self.assertEqual(len(self.json[DATASETS]['tr05_real']), 1600,
                         f'Dataset tr05_real should contain 1600 examples')
        self.assertEqual(len(self.json[DATASETS]['dt05_simu']), 1640,
                         f'Dataset dt05_simu should contain 1640 examples')
        self.assertEqual(len(self.json[DATASETS]['dt05_real']), 1640,
                         f'Dataset dt05_real should contain 1640 examples')
        self.assertEqual(len(self.json[DATASETS]['et05_simu']), 1320,
                         f'Dataset et05_simu should contain 1320 examples')
        self.assertEqual(len(self.json[DATASETS]['et05_real']), 1320,
                         f'Dataset et05_real should contain 1320 examples')


if __name__ == '__main__':
    unittest.main()
