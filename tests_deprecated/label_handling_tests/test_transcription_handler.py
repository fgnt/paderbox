import tempfile
import unittest
from os import path

import paderbox.testing as tc
from paderbox.label_handling.transcription_handler import WordTranscriptionHandler


class TestTransHandler(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"ab": ["a", "b"], "c": ["c"], "def": ["d", "e", "f"]}

    def test_length(self):
        mandatory_tok = ["blank"]
        th = WordTranscriptionHandler(
            lexicon=self.lexicon, labels=mandatory_tok, oov_word="<unk>")
        tokens = {token for tokens in self.lexicon.values() for token in tokens}
        num_tokens = len(tokens) + len(mandatory_tok) + 1
        self.assertEqual(num_tokens, len(th.labels))
        self.assertEqual(len(th.int_to_label), len(th.label_to_int))

    def test_mandatory(self):
        mandatory_tok = ["eps", "phi", "blank"]
        th = WordTranscriptionHandler(
            lexicon=self.lexicon, labels=mandatory_tok, oov_word="<unk>")
        self.assertEqual(
            [th.int_to_label[idx] for idx in range(len(mandatory_tok))],
            mandatory_tok)

    def test_mapping(self):
        th = WordTranscriptionHandler(self.lexicon, oov_word="<unk>")

        word_seq = ["ab", "c", "def", "g"]
        label_seq = th.prepare_transcription(word_seq)

        expected = ["a", "b", "c", "d", "e", "f", "<unk>"]
        self.assertEqual(expected, label_seq)

    def test_mapping_to_int(self):
        th = WordTranscriptionHandler(lexicon=self.lexicon, labels=["<blank>"])

        word_seq = ["ab", "c", "def"]
        label_seq = th.process_transcription(word_seq)

        expected = range(1, 7)  # expecting a sorted label mapping here
        tc.assert_array_equal(expected, label_seq)

    def test_case_insensitive(self):
        th = WordTranscriptionHandler(self.lexicon, case_sensitive=False)
        self.assertEqual(
            th.lexicon, {"AB": ["a", "b"], "C": ["c"], "DEF": ["d", "e", "f"]})

        word_seq = ["AB", "c", "dEf"]
        label_seq = th.prepare_transcription(word_seq)
        expected = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(expected, label_seq)

    def test_save_and_load(self):
        th_orig = WordTranscriptionHandler(self.lexicon, eow="<eow>")
        with tempfile.TemporaryDirectory() as _dir:
            file = path.join(_dir, "th.json")
            th_orig.save(file)
            th_load = WordTranscriptionHandler.load(file)
        self.assertEqual(th_orig.__dict__, th_load.__dict__)


if __name__ == '__main__':
    unittest.main()
