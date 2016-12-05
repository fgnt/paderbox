import json
import unittest

from nt.testing import db_test
from nt.io import load_json

ger_json = db_test.ROOT / "german.json"
#ger_json = "german_speechdata_package_v2.json"

class test_ger_db(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(ger_json)

    def test_annot(self):
        self.assertIn("annotations", self.json)


if __name__ == '__main__':
    unittest.main()