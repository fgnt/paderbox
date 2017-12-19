import unittest

from nt.database.keys import *
from nt.io import load_json
from nt.testing import db_test

merl = db_test.ROOT / "merl_speaker_mixtures_min_8k.json"


class TestMerlDatabase(db_test.DatabaseTest):
    def setUp(self):
        self.json = load_json(merl)

    def test_dataset(self):
        self.assert_in_datasets(['mix_2_spk_min_cv'])

    def test_len(self):
        # total length
        self.assert_total_length(56000)

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
        self.assert_in_example([TRANSCRIPTION, AUDIO_PATH])


if __name__ == '__main__':
    unittest.main()
