import unittest

from nt.io.data_dir import database_jsons as database_jsons_dir

ROOT = database_jsons_dir

class DatabaseTest(unittest.TestCase):

    def _test_train_test_dev(self):
        self.assertIn("test", self.json)
        self.assertIn("train", self.json)
        self.assertIn("dev", self.json)

    def _test_orth(self):
        self.assertIn("orth", self.json)

    def _test_flists(self):
        self.assertIn("flists", self.json)

