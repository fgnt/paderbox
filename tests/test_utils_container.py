from nt.utils import Container
import unittest

class ContainerTest(unittest.TestCase):

    def setUp(self):
        self.c = Container()

    def test_add_items(self):
        self.c.a1 = 1
        self.c['a2'] = 2
        self.assertEqual(self.c.a1, 1)
        self.assertEqual(self.c.a2, 2)
        self.assertEqual(self.c['a1'], 1)
        self.assertEqual(self.c['a2'], 2)

    def test_rm_item(self):
        self.c.a1 = 3
        del self.c.a1
        self.assertFalse(hasattr(self.c, 'a1'))