import unittest

from nt.database.keys import *
from nt.io import load_json
from nt.testing import db_test

merl = db_test.ROOT / "merl_speaker_mixtures_min_8k.json"


class TestMerlDatabase(db_test.DatabaseTest):
    def setUp(self):
        self.json = load_json(merl)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)
        self.assertIn('mix_2_spk_min_cv', self.json[DATASETS])

    def test_len(self):
        ids = 56000
        self.assertEqual(
            len(list(self.json[EXAMPLES])),
            ids,
            f"There should be {ids} utt_ids in '{EXAMPLES}'!"
        )
        datasets = 6
        self.assertEqual(
            len(list(self.json[DATASETS])),
            datasets,
            f"There should be {datasets} datasets in '{DATASETS}'!"
        )
        dataset_length = dict(
            mix_2_spk_min_cv=5000
        )
        for k, v in dataset_length:
            self.assertEqual(
                len(list(self.json[DATASETS][k])),
                v, f"There should be {v} utt_ids in dataset '{k}'!"
            )

    def test_examples(self):
        utt_id = list(self.json[EXAMPLES])[0]
        # audio_path
        self.assertIn(AUDIO_PATH, self.json[EXAMPLES][utt_id])
        # transcription
        self.assertIn(TRANSCRIPTION, self.json[EXAMPLES][utt_id])


if __name__ == '__main__':
    unittest.main()
