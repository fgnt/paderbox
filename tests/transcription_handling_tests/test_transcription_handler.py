import unittest
from nt.transcription_handling.transcription_handler import TranscriptionHandler
import nt.testing as tc


class TestTransHandler(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"ab": ["a", "b"], "c": ["c"], "def": ["d", "e", "f"]}

    def test_length(self):
        mandatory_tok = ["blank"]
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=mandatory_tok,
                                  oov_word="<unk>")
        tokens = {token for tokens in self.lexicon.values() for token in tokens}
        num_tokens = len(tokens) + len(mandatory_tok) + 1
        self.assertEqual(num_tokens, len(th.labels))
        self.assertEqual(len(th.int_to_label), len(th.label_to_int))

    def test_mandatory(self):
        mandatory_tok = ["eps", "phi", "blank"]
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=mandatory_tok,
                                  oov_word="<unk>")
        self.assertEqual(
            [th.int_to_label[idx] for idx in range(len(mandatory_tok))],
            mandatory_tok)

    def test_mapping(self):
        th = TranscriptionHandler(self.lexicon, oov_word="<unk>")

        word_seq = ["ab", "c", "def", "g"]
        label_seq = th.prepare_target(word_seq, to_int=False)

        expected = ["a", "b", "c", "d", "e", "f", "<unk>"]
        self.assertEqual(expected, label_seq)

    def test_mapping_to_int(self):
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=["<blank>", ])

        word_seq = ["ab", "c", "def"]
        label_seq = th.prepare_target(word_seq)

        expected = range(1, 7)  # expecting a sorted label mapping here
        tc.assert_array_equal(expected, label_seq)

    def test_case_insensitive(self):
        th = TranscriptionHandler(self.lexicon, case_sensitive=False)
        self.assertEqual(
            th.lexicon, {"AB": ["a", "b"], "C": ["c"], "DEF": ["d", "e", "f"]})

        word_seq = ["AB", "c", "dEf"]
        label_seq = th.prepare_target(word_seq, to_int=False)
        expected = ["a", "b", "c", "d", "e", "f"]
        self.assertEqual(expected, label_seq)

if __name__ == '__main__':
    unittest.main()
