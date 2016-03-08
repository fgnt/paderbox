__author__ = 'walter'
import unittest
import tempfile
from nt.speech_recognition.lexicon import Linear
from nt.transcription_handling.transcription_handler import TranscriptionHandler
from nt.speech_recognition import fst

class TestLexicon(unittest.TestCase):

    def setUp(self):
#        self.words = ['Mars', 'man', 'Martian', 'Marsman']
        self.words = ['AA', 'AB']
        self.lexicon = {word: list(word) for word in self.words}
        self.special_symbols = {'blank': None, 'eps': '<eps>', 'phi': '<phi>', 'sow': '<sow>',
                                'eow': '</eow>', 'sos': '<sos>', 'eos': '</eos>'}

        self.transcription_handler = TranscriptionHandler(self.lexicon, **self.special_symbols)
        self.transcription_handler.add_word(self.special_symbols['eos'], [self.special_symbols['eos']])

        self.word_offset = len(self.transcription_handler.label_handler)
        self.int_lexicon = {self.transcription_handler.words2ints(word)[0] + self.word_offset:
                            self.transcription_handler.labels2ints(labels) for word, labels in self.lexicon.items()}
        self.int_eos_word = self.transcription_handler.words2ints(self.special_symbols['eos'])[0] + self.word_offset

        self.int_eps = self.transcription_handler.labels2ints(self.special_symbols['eps'])[0]
        self.int_eow = self.transcription_handler.labels2ints(self.special_symbols['eow'])[0]
        self.int_phi = self.transcription_handler.labels2ints(self.special_symbols['phi'])[0]
        self.int_sow = self.transcription_handler.labels2ints(self.special_symbols['sow'])[0]
        self.int_eos_label = self.transcription_handler.labels2ints(self.special_symbols['eos'])[0]
        self.int_label = [self.transcription_handler.labels2ints(label)[0]
                          for label in self.transcription_handler.label_handler.labels]

    def _test_write_fst(self, class_type, add_word_mode, build_character_model_mode):
        lexicon = class_type(self.int_eps, self.int_eow)
        for word in self.int_lexicon.items():
            lexicon.add_word(word, add_word_mode)

        lexicon.build_character_model(self.int_label, build_character_model_mode, self.int_sow)
        lexicon.add_eos(self.int_eos_label, self.int_eos_word)
        lexicon.add_self_loops(self.int_phi)

        with tempfile.TemporaryDirectory() as directory:
            fst_filename = directory + '/L.fst'
            sym_filename = fst_filename + '.syms'
            pdf_filename = fst_filename + '.pdf'
            lexicon.write_fst(fst_filename)
            self.transcription_handler.write_table(sym_filename)
            fst.draw(sym_filename, sym_filename, fst_filename, pdf_filename)

    def test_linear_linear_trie_write_fst(self):
        self._test_write_fst(Linear, 'linear', 'trie')

    def test_linear_linear_flat_write_fst(self):
        self._test_write_fst(Linear, 'linear', 'flat')

    def test_linear_linear_copy_write_fst(self):
        self._test_write_fst(Linear, 'linear', 'copy')

    def test_linear_linear_linear_write_fst(self):
        self._test_write_fst(Linear, 'linear', 'linear')

    def test_linear_trie_trie_write_fst(self):
        self._test_write_fst(Linear, 'trie', 'trie')

    def test_linear_trie_flat_write_fst(self):
        self._test_write_fst(Linear, 'trie', 'flat')

    def test_linear_trie_copy_write_fst(self):
        self._test_write_fst(Linear, 'trie', 'copy')

    def test_linear_trie_linear_write_fst(self):
        self._test_write_fst(Linear, 'trie', 'linear')
