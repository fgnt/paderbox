import json
import unittest

from nt.testing import db_test

tidigits_json = db_test.ROOT + "/tidigits.json"
#tidigits_json = "tidigits.json"

class test_tidigits_db(db_test.DatabaseTest):

    def setUp(self):
        with open(tidigits_json) as file:
            self.json = json.load(file)

    def test_train_test_dev(self):
        self.assertIn("test", self.json)
        self.assertIn("train", self.json)

    def test_dict(self):
        self.assertIn("dict", self.json)



if __name__ == '__main__':
    unittest.main()