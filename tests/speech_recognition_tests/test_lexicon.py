__author__ = 'walter'
import unittest
import tempfile
from nt.speech_recognition.lexicon import Trie
from nt.speech_recognition.lexicon import Linear
from nt.transcription_handling.transcription_handler import TranscriptionHandler
from nt.speech_recognition import fst

class TestLexicon(unittest.TestCase):

    def setUp(self):
#        self.words = ['Mars', 'man', 'Martian', 'Marsman']
        self.words = ['AA', 'AB', 'AB']
        self.lexicon = {word: list(word) for word in self.words}

        self.special_symbols = {'blank': '<blank>', 'eps': '<eps>', 'phi': '<phi>', 'sow': '<sow>',
                                'eow': '</eow>', 'sos': '<sos>', 'eos': '</eos>'}
        self.transcription_handler = TranscriptionHandler(self.lexicon, **self.special_symbols)

        self.word_offset = len(self.transcription_handler.label_handler)
        self.int_lexicon = {self.transcription_handler.words2ints(word)[0] + self.word_offset:
                            self.transcription_handler.labels2ints(labels) for word, labels in self.lexicon.items()}
        self.int_eps = self.transcription_handler.labels2ints(self.special_symbols['eps'])[0]
        self.int_eos = self.transcription_handler.labels2ints(self.special_symbols['eow'])[0]
        self.int_label = [self.transcription_handler.labels2ints(label)[0]
                          for label in self.transcription_handler.label_handler.labels]


    def test_trie_write_fst(self):
        word_trie = Trie(self.int_eps, self.int_eos)
        for word in self.int_lexicon.items():
            word_trie.add_word(word)

        word_trie.build_character_model(self.int_label)

        with tempfile.TemporaryDirectory() as directory:
            fst_filename = directory + '/L.fst'
            sym_filename = fst_filename + '.syms'
            pdf_filename = fst_filename + '.pdf'
            word_trie.write_fst(fst_filename)
            self.transcription_handler.write_table(sym_filename)
            fst.draw(sym_filename, sym_filename, fst_filename, pdf_filename)


    def test_linear_add_words(self):
        word_linear = Linear(self.int_eps, self.int_eos)
        word_linear.add_words(self.int_lexicon)

        word_linear.build_language_model(self.int_label)

        with tempfile.TemporaryDirectory() as directory:
            fst_filename = directory + '/L.fst'
            sym_filename = fst_filename + '.syms'
            pdf_filename = fst_filename + '.pdf'
            word_linear.write_fst(fst_filename)
            self.transcription_handler.write_table(sym_filename)
            fst.draw(sym_filename, sym_filename, fst_filename, pdf_filename)
