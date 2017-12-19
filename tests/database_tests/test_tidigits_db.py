import unittest

from nt.testing import db_test
from nt.io import load_json
from nt.database.keys import *

tidigits_json = db_test.ROOT / "tidigits.json"


class TestTidigitsDatabase(db_test.DatabaseTest):

    def setUp(self):
        self.json = load_json(tidigits_json)

if __name__ == '__main__':
    unittest.main()
