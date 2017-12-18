import json
import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

tidigits_json = db_test.ROOT / "tidigits.json"

class test_tidigits_db(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(tidigits_json)

    def test_structure(self):
        self.assertIn(DATASETS, self.json)
        self.assertIn(EXAMPLES, self.json)
        self.assertGreater(len(list(self.json[EXAMPLES])), 1)

if __name__ == '__main__':
    unittest.main()
