import unittest
from nt.transcription_handling.transcription_handler import TranscriptionHandler
import nt.testing as tc


class TestTransHandler(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"ab": ["a", "b"], "c": ["c"], "def": ["d", "e", "f"]}

    def test_mapping(self):
        th = TranscriptionHandler(self.lexicon, oov_word="<unk>")

        word_seq = ["ab", "c", "def", "g"]
        label_seq = th.transcription_to_labels(word_seq, map_to_int=False)

        expected = ["a", "b", "c", "d", "e", "f", "<unk>"]
        self.assertEqual(expected, label_seq)

    def test_mapping_to_int(self):
        th = TranscriptionHandler(self.lexicon)

        word_seq = ["ab", "c", "def"]
        label_seq = th.transcription_to_labels(word_seq)

        expected = range(1, 7)  # expecting a sorted label mapping here
        tc.assert_array_equal(expected, label_seq)

    def test_add_eps(self):
        th1 = TranscriptionHandler(self.lexicon, oov_word="<unk>")
        th2 = th1.add_eps()

        word_seq = ["ab", "c", "def"]
        label_seq_1 = th1.transcription_to_labels(word_seq)
        label_seq_2 = th2.transcription_to_labels(word_seq)

        tc.assert_array_equal(label_seq_1 + 1, label_seq_2)

    def test_disambs(self):
        lexicon = {"test": ["test"], "testcase": ["test", "case"]}
        th = TranscriptionHandler(lexicon)
        th.add_disambs()

        expected = ["test", "#1"]
        self.assertEqual(expected, th.lexicon["test"])

    def test_clean_up(self):
        th = TranscriptionHandler(self.lexicon)
        words = {"ab", "c"}
        th.clean_up_lexicon(words)

        self.assertEqual(words, set(th.lexicon.keys()))

if __name__ == '__main__':
    unittest.main()
