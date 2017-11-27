import unittest
import numpy as np
from nt.label_handling.label_handler import LabelHandler
import nt.testing as tc
import tempfile
from os import path


class TestLabelHandler(unittest.TestCase):
    def setUp(self):
        self.labels = ['a', 'b', 'c']
        self.lh = LabelHandler(self.labels)

    def test_add(self):
        new_labels = ['d', 'e']
        self.lh.add_labels(new_labels)
        self.assertEqual(self.lh.num_labels, 5)
        self.assertEqual(self.lh.labels, self.labels + new_labels)
        self.assertEqual(self.lh.label_to_int['e'], 4)
        self.assertEqual(self.lh.int_to_label[4], 'e')

    def test_mapping(self):
        self.assertEqual(self.lh.labels2ints(
            self.labels, toarray=False), list(range(3)))
        self.assertEqual(self.lh.ints2labels(range(3)), self.labels)

    def test_segmentation(self):
        segmentation = [('a', 0, 3), ('c', 1, 3), ('b', 0, 1)]
        target = self.lh.process_segmentation(segmentation)
        self.assertEqual(target.tolist(), [[1, 1, 0,],[1, 0, 1],[1, 0, 1]])
        target = self.lh.process_segmentation(segmentation, seq_len=4)
        self.assertEqual(
            target.tolist(), [[1, 1, 0,],[1, 0, 1],[1, 0, 1],[0, 0, 0]])

    def test_tags(self):
        tags = ['b', 'a']
        target = self.lh.process_tags(tags)
        self.assertEqual(target.tolist(), [1, 1, 0,])


if __name__ == '__main__':
    unittest.main()
