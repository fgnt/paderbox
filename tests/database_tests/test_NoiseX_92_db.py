import unittest
import json
import db_test

NoiseX_92_json = db_test.ROOT + "/noisex_92.json"
# NoiseX_92_json = "NoiseX_92.json"


class test_NoiseX_92_db(db_test.DatabaseTest):

        def setUp(self):
            with open(NoiseX_92_json) as file:
                self.json = json.load(file)

        def test_train_test_dev(self):
            self.assertIn("train", self.json)

        def test_orth(self):
            self.assertNotIn("orth", self.json)

        def test_set_len(self):
            self.assertEqual(len(self.json['train']['flists']['wav']['standard set']), 15)
            self.assertEqual(len(self.json['train']['flists']['wav']['16kHz set']), 18)
            self.assertEqual(len(self.json['train']['flists']['wav']['metro set']), 1)


if __name__ == '__main__':
    unittest.main()