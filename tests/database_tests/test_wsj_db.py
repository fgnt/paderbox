import unittest
import json
import db_test

wsj_json = db_test.ROOT + "/wsj.json"
# wsj_json = "wsj.json"

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

class test_wsj_db(db_test.DatabaseTest):

    def setUp(self):
        with open(wsj_json) as file:
            self.json = json.load(file)

    def test_official_len(self):
        self.assertEqual(len(self.json['train']['flists']['wave']['official_si_284']), 37416,
                         'Warning: expected 37416 lines in official_si_284.flist')
        self.assertEqual(len(self.json['train']['flists']['wave']['official_si_84']), 7138,
                         'Warning: expected 7138 lines in official_si_84.flist')
        self.assertEqual(len(self.json['test']['flists']['wave']['official_si_et_20']), 333,
                         'Warning: expected 333 lines in official_si_et_20.flist')
        self.assertEqual(len(self.json['test']['flists']['wave']['official_si_et_05']), 330,
                         'Warning: expected 330 lines in official_si_et_05.flist')
        self.assertEqual(len(self.json['test']['flists']['wave']['official_si_et_h1/wsj64k']), 213,
                         'Warning: expected 213 lines in official_si_et_h1/wsj64k.flist')
        self.assertEqual(len(self.json['test']['flists']['wave']['official_si_et_h2/wsj5k']), 215,
                         'Warning: expected 213 lines in official_si_et_h2/wsj5k.flist')  # Should be 213 not 215 links!
        self.assertEqual(len(self.json['test']['flists']['wave']['official_si_dt_20']), 503,
                         'Warning: expected 503 lines in official_si_dt_20flist')
        self.assertEqual(len(self.json['dev']['flists']['wave']['official_si_dt_05']), 513,
                         'Warning: expected 513 lines in official_si_dt_05.flist')
        self.assertEqual(len(self.json['dev']['flists']['wave']['official_Dev-set_Hub_1']), 503,
                         'Warning: expected 503 lines in Dev-set_Hub_1.flist')
        self.assertEqual(len(self.json['dev']['flists']['wave']['official_Dev-set_Hub_2']), 913,
                         'Warning: expected 913 lines in Dev-set_Hub_2.flist')

    def test_official_words(self):
        word = self.json['orth']["word"]
        self.assertEqual(complete(self.json['train']['flists']['wave']['official_si_284'], word), 1)
        self.assertEqual(complete(self.json['train']['flists']['wave']['official_si_84'], word), 1)
        self.assertEqual(complete(self.json['test']['flists']['wave']['official_si_et_20'], word), 1)
        self.assertEqual(complete(self.json['test']['flists']['wave']['official_si_et_05'], word), 1)
        self.assertEqual(complete(self.json['test']['flists']['wave']['official_si_et_h1/wsj64k'], word), 1)
        self.assertEqual(complete(self.json['test']['flists']['wave']['official_si_et_h2/wsj5k'], word), 1)
        self.assertEqual(complete(self.json['test']['flists']['wave']['official_si_dt_20'], word), 1)
        self.assertEqual(complete(self.json['dev']['flists']['wave']['official_si_dt_05'], word), 1)
        self.assertEqual(complete(self.json['dev']['flists']['wave']['official_Dev-set_Hub_1'], word), 1)
        self.assertEqual(complete(self.json['dev']['flists']['wave']['official_Dev-set_Hub_2'], word), 1)


if __name__ == '__main__':
    unittest.main()

