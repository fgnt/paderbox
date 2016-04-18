import json
import unittest

from nt.testing import db_test

ger_json = db_test.ROOT + "/german.json"
#ger_json = "german_speechdata_package_v2.json"

class test_ger_db(db_test.DatabaseTest):

    def setUp(self):
        with open(ger_json, encoding='utf-8') as file:
            self.json = json.load(file)

    def test_annot(self):
        self.assertIn("annotations", self.json)


if __name__ == '__main__':
    unittest.main()