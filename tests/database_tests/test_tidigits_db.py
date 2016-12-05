import json
import unittest

from nt.testing import db_test
from nt.io import load_json

tidigits_json = db_test.ROOT / "tidigits.json"
#tidigits_json = "tidigits.json"

class test_tidigits_db(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(tidigits_json)

    def test_train_test_dev(self):
        self.assertIn("test", self.json)
        self.assertIn("train", self.json)

    def test_dict(self):
        self.assertIn("dict", self.json)



if __name__ == '__main__':
    unittest.main()
