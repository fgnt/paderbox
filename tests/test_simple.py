import unittest

class TestSimple(unittest.TestCase):
    
    def test_failure(self):
        self.assertTrue(False)

    def test_success(self):
        self.assertTrue(True)
