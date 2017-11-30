import unittest
import numpy as np
from nt.label_handling.label_handler import LabelHandler
from nt.database import keys
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
        self.assertEqual(self.lh._label_to_int['e'], 4)
        self.assertEqual(self.lh._int_to_label[4], 'e')

    def test_transcription(self):
        self.assertEqual(self.lh.process_transcription(
            self.labels, toarray=False), list(range(3)))
        self.assertEqual(self.lh.recover_transcription(range(3)), self.labels)

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

    def test_src_mapping(self):
        lh = LabelHandler(['a', 'b'], src_mapping={'c': 'b'})
        self.assertEqual(lh.process_transcription(
            ['a', 'b', 'c'], toarray=False), [0, 1, 1])

    def test_oov(self):
        lh = LabelHandler(['a', 'c',], oov_label='<unk>')
        self.assertEqual(lh.num_labels, 3)
        self.assertEqual(lh.process_transcription(
            ['a', 'b', 'c'], toarray=False), [0, 2, 1])

    def test_automatic_add(self):
        lh = LabelHandler(oov_label='<unk>', freeze=False)
        self.assertEqual(lh.num_labels, 1)
        self.assertEqual(lh.process_transcription(
            ['a', 'b', 'c'], toarray=False), [1, 2, 3])
        self.assertEqual(lh.num_labels, 4)
        lh.freeze()
        self.assertEqual(lh.process_transcription(
            ['c', 'd',], toarray=False), [3, 0,])
        self.assertEqual(lh.num_labels, 4)

    def test_read(self):
        lh = LabelHandler(label_type='chars')
        examples = [
            {
                keys.TRANSCRIPTION: {'chars': ['a', 'b',]},
                keys.TAGS: {'chars': ['c', 'd',]},
                keys.SEGMENTATION: {'chars': [('e', 0, 2), ('f', 1, 4)]}
            }
        ]
        self.assertEqual(lh.num_labels, 0)
        lh.read_transcriptions(examples)
        self.assertEqual(lh.num_labels, 2)
        lh.read_segmentations(examples)
        self.assertEqual(lh.num_labels, 4)
        lh.read_tags(examples)
        self.assertEqual(lh.num_labels, 6)



if __name__ == '__main__':
    unittest.main()
