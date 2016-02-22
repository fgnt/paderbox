import unittest
import json
import db_test

NoiseX_92_json = db_test.ROOT + "NoiseX_92.json"
# NoiseX_92_json = "NoiseX_92.json"


class test_NoiseX_92_db(db_test.DatabaseTest):

        def setUp(self):
            with open(NoiseX_92_json) as file:
                self.json = json.load(file)

        def test_train_test_dev(self):
            self.assertIn("test_train", self.json)

        def test_orth(self):
            self.assertNotIn("orth", self.json)

        def test_set_len(self):
            self.assertEqual(len(self.json['test_train']['complete set']['wav']), 34)
            self.assertEqual(len(self.json['test_train']['standart set']['wav']), 15)
            self.assertEqual(len(self.json['test_train']['16kHz set']['wav']), 18)
            self.assertEqual(len(self.json['test_train']['metro set']['wav']), 1)