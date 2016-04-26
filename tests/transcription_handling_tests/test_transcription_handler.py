import unittest
from nt.transcription_handling.transcription_handler import TranscriptionHandler
import nt.testing as tc
from copy import deepcopy


class TestTransHandler(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"ab": ["a", "b"], "c": ["c"], "def": ["d", "e", "f"]}

    def test_length(self):
        mandatory_tok = ["blank"]
        mandatory_words = ["test"]
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=mandatory_tok,
                                  mandatory_words=mandatory_words,
                                  oov_word="<unk>")
        tokens = {token for tokens in self.lexicon.values() for token in tokens}
        num_tokens = len(tokens) + len(mandatory_tok) + 1
        num_words = len(self.lexicon) + len(mandatory_words) + 1
        self.assertEqual(num_tokens, len(th.tokens))
        self.assertEqual(num_words, len(th.words))
        num_labels = len(set(th.tokens).union(th.words))
        self.assertEqual(num_labels, len(th.label_to_int))
        self.assertEqual(len(th.int_to_label), len(th.label_to_int))

    def test_mandatory(self):
        mandatory_tok = ["eps", "phi", "blank"]
        mandatory_words = ["word1", "word2"]
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=mandatory_tok,
                                  mandatory_words=mandatory_words,
                                  oov_word="<unk>")
        self.assertEqual(
            [th.int_to_label[idx] for idx in range(len(mandatory_tok))],
            mandatory_tok)
        self.assertEqual(
            [th.int_to_label[idx + th.num_tokens]
             for idx in range(len(mandatory_words))],
            mandatory_words)

    def test_word_offset(self):
        mandatory_tokens = ["token1", "token2"]
        mandatory_words = ["word1", "word2"]
        word_off = 100
        th = TranscriptionHandler(self.lexicon,
                                  mandatory_tokens=mandatory_tokens,
                                  mandatory_words=mandatory_words,
                                  word_offset=word_off, oov_word="<unk>")

        self.assertEqual([th.label_to_int[token] for token in mandatory_tokens],
                         [idx for idx in range(len(mandatory_tokens))])
        self.assertEqual([th.label_to_int[word] for word in mandatory_words],
                         [idx+word_off for idx in range(len(mandatory_words))])

    def test_mapping(self):
        th = TranscriptionHandler(self.lexicon, oov_word="<unk>")

        word_seq = ["ab", "c", "def", "g"]
        label_seq = th.words2tokens(word_seq, map_to_int=False)

        expected = ["a", "b", "c", "d", "e", "f", "<unk>"]
        self.assertEqual(expected, label_seq)

    def test_mapping_to_int(self):
        th = TranscriptionHandler(self.lexicon, mandatory_tokens=["<blank>", ])

        word_seq = ["ab", "c", "def"]
        label_seq = th.words2tokens(word_seq)

        expected = range(1, 7)  # expecting a sorted label mapping here
        tc.assert_array_equal(expected, label_seq)

    def test_add_eps(self):
        th1 = TranscriptionHandler(self.lexicon, oov_word="<unk>")
        th2 = deepcopy(th1)
        th2.insert_token("<eps>", 0)

        word_seq = ["ab", "c", "def"]
        label_seq_1 = th1.words2tokens(word_seq)
        label_seq_2 = th2.words2tokens(word_seq)

        tc.assert_array_equal(label_seq_1 + 1, label_seq_2)

    def test_disambs(self):
        lexicon = {"test": ["test"], "testcase": ["test", "case"]}
        th = TranscriptionHandler(lexicon)
        th.add_disambigs()

        expected = ["test", "#1"]
        self.assertEqual(expected, th.lexicon["test"])

    def test_clean_up(self):
        th = TranscriptionHandler(self.lexicon)
        words = {"ab", "c"}
        th.clean_up_lexicon(words)

        self.assertEqual(words, set(th.lexicon.keys()))

if __name__ == '__main__':
    unittest.main()
