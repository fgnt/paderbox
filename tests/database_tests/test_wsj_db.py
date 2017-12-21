import unittest

from nt.io import load_json
from nt.testing import db_test
from nt.database.keys import *

wsj_json = db_test.ROOT / "wsj.json"


def complete(scenario, word):
    """ Checks whether there is a word transcription of every utterance ID in a give scenario

    :param scenario: the scenario that has to be controlled
    :param word: the word transcriptions of all sentences
    :return: nothing to return, prints out the utterance IDs, that have no word translations
    """

    for key in scenario:
        if key.lower() not in word:
            return 0
        return 1


class TestWSJDatabase(db_test.DatabaseTest):
    def setUp(self):
        self.json = load_json(wsj_json)

    def test_examples(self):
        self.assert_in_example([TRANSCRIPTION, AUDIO_PATH])

    def test_len(self):
        self.assert_total_length(46661)
        self.assertEqual(
            len(self.json[DATASETS]['train_si284']), 37416,
            'Warning: expected 37416 lines in train_si284')
        self.assertEqual(
            len(self.json[DATASETS]['train_si84']), 7138,
            'Warning: expected 7138 lines in train_si84')
        self.assertEqual(
            len(self.json[DATASETS]['test_eval92']), 333,
            'Warning: expected 333 lines in test_eval92')
        self.assertEqual(
            len(self.json[DATASETS]['test_eval92_5k']), 330,
            'Warning: expected 330 lines in test_eval92_5k')
        self.assertEqual(len(self.json[DATASETS]['test_eval93']), 213,
                         'Warning: expected 213 lines in test_eval93')
        self.assertEqual(
            len(self.json[DATASETS]['test_eval93_5k']), 215,
            'Warning: expected 215 lines in test_eval93_5k')
        self.assertEqual(
            len(self.json[DATASETS]['cv_dev93']), 503,
            'Warning: expected 503 lines in cv_dev93')
        self.assertEqual(
            len(self.json[DATASETS]['cv_dev93_5k']), 513,
            'Warning: expected 513 lines in cv_dev93_5k')


    # def test_official_words(self):
    #     word = self.json['orth']["word"]
    #     self.assertEqual(
    #         complete(self.json['train']['flists']['wave']['official_si_284'],
    #                  word), 1)
    #     self.assertEqual(
    #         complete(self.json['train']['flists']['wave']['official_si_84'],
    #                  word), 1)
    #     self.assertEqual(
    #         complete(self.json['test']['flists']['wave']['official_si_et_20'],
    #                  word), 1)
    #     self.assertEqual(
    #         complete(self.json['test']['flists']['wave']['official_si_et_05'],
    #                  word), 1)
    #     self.assertEqual(complete(
    #         self.json['test']['flists']['wave']['official_si_et_h1/wsj64k'],
    #         word), 1)
    #     self.assertEqual(complete(
    #         self.json['test']['flists']['wave']['official_si_et_h2/wsj5k'],
    #         word), 1)
    #     self.assertEqual(
    #         complete(self.json['dev']['flists']['wave']['official_si_dt_20'],
    #                  word), 1)
    #     self.assertEqual(
    #         complete(self.json['dev']['flists']['wave']['official_si_dt_05'],
    #                  word), 1)
    #     self.assertEqual(complete(
    #         self.json['dev']['flists']['wave']['official_Dev-set_Hub_1'], word),
    #                      1)
    #     self.assertEqual(complete(
    #         self.json['dev']['flists']['wave']['official_Dev-set_Hub_2'], word),
    #                      1)


if __name__ == '__main__':
    unittest.main()
