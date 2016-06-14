import unittest
from nt.transcription_handling.transcription_handler import TranscriptionHandler
import nt.testing as tc
from copy import deepcopy


class TestTransHandler(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"ab": ["a", "b"], "c": ["c"], "def": ["d", "e", "f"]}

    def test_length(self):
        mandatory_tok = ["blank"]
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=mandatory_tok,
                                  oov_word="<unk>", add_words_from_lexicon=True)
        tokens = {token for tokens in self.lexicon.values() for token in tokens}
        num_tokens = len(tokens) + len(mandatory_tok) + 1
        num_words = len(self.lexicon) + 1
        self.assertEqual(num_tokens, len(th.tokens))
        self.assertEqual(num_words, len(th.words))
        num_labels = len(set(th.tokens).union(th.words))
        self.assertEqual(num_labels, len(th.label_to_int))
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

    def test_add_eps(self):
        th1 = TranscriptionHandler(self.lexicon, oov_word="<unk>")
        th2 = deepcopy(th1)
        th2.insert_token("<eps>", 0)

        word_seq = ["ab", "c", "def"]
        label_seq_1 = th1.prepare_target(word_seq)
        label_seq_2 = th2.prepare_target(word_seq)

        tc.assert_array_equal(label_seq_1 + 1, label_seq_2)

if __name__ == '__main__':
    unittest.main()
