import unittest
import json
import db_test

timit_json = db_test.ROOT + "/timit.json"
#timit_json = "TIMIT.json"

class test_timit_db(db_test.DatabaseTest):

    def setUp(self):
        with open(timit_json) as file:
            self.json = json.load(file)

    def test_train_test_dev(self):
        self.assertIn("test", self.json)
        self.assertIn("train", self.json)
        #self.assertIn("dev", self.json) # no dev in timit.json

    def test_orth(self):
        self.assertIn("orth", self.json)
        self.assertIn("annotation", self.json)

    def test_Complete_databank_len(self):
        self.assertEqual(len(self.json['test']['Complete Test Set']['wav']), 1680,
                         "There should be 1680 files in 'Complete Test Set'!")
        self.assertEqual(len(self.json['train']['Complete Train Set']['wav']), 4620,
                         "There should be 4620 files in 'Complete Train Set'!")

    def test_Core_Test_Set_len(self):
        self.assertEqual(len(self.json['test']['Core Test Set']['wav']),192,
                         "There should be 192 files in 'Core Test Set'!")

    def test_orth_len(self):
        phoneme_len = len(self.json['orth']['phoneme'])
        word_len = len(self.json['orth']['word'])
        self.assertEqual(phoneme_len, 6300, "Databank has wrong number of phnome transciprions!")
        self.assertEqual(word_len, 6300, "Databank has wrong number of word transciprions!")

    def test_flists_len(self):
        self.assertEqual(len(self.json['flists']), 6492)

if __name__ == '__main__':
    unittest.main()