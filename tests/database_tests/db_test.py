import unittest

ROOT = '/net/storage/database_jsons/'

class DatabaseTest(unittest.TestCase):

    def test_train_test_dev(self):
        self.assertIn("test", self.json)
        self.assertIn("train", self.json)
        self.assertIn("dev", self.json)

    def test_orth(self):
        self.assertIn("orth", self.json)

    def test_flists(self):
        self.assertIn("flists", self.json)

