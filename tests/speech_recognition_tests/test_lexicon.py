
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
        self.special_symbols = {'eps': '<eps>', 'phi': '<phi>',
                                'sow': '<sow>', 'eow': '</eow>',
                                'eos': '</eos>', 'eoc': '</eoc>'}
        mandatory_tokens = [self.special_symbols['eps'],
                            self.special_symbols['phi'], ]
        self.th = TranscriptionHandler(self.lexicon,
                                       mandatory_tokens=mandatory_tokens)
        self.th.add_tokens(self.special_symbols.values())
        self.th.add_words(
            {self.special_symbols['eos']: [self.special_symbols['eos']]})

        self.int_lexicon = {self.th.labels2ints(word)[0]:
                            self.th.labels2ints(labels)
                            for word, labels in self.lexicon.items()}
        self.int_eos_word = self.th.labels2ints(
            self.special_symbols['eos'])[0]

        self.int_eps = self.th.labels2ints(
            self.special_symbols['eps'])[0]
        self.int_eow = self.th.labels2ints(
            self.special_symbols['eow'])[0]
        self.int_phi = self.th.labels2ints(
            self.special_symbols['phi'])[0]
        self.int_sow = self.th.labels2ints(
            self.special_symbols['sow'])[0]
        self.int_eoc = self.th.labels2ints(
            self.special_symbols['eoc'])[0]
        self.int_eos_label = self.th.labels2ints(
            self.special_symbols['eos'])[0]
        self.int_label = [self.th.labels2ints(label)[0]
                          for label in self.th.tokens
                          if label not in self.special_symbols.values()]

    def _test_write_fst(self, class_type, add_word_mode,
                        build_character_model_mode, sow=None, eoc=None):
        lexicon = class_type(self.int_eps, self.int_eow, eoc)
        for word in self.int_lexicon.items():
            lexicon.add_word(word, add_word_mode)

        lexicon.build_character_model(self.int_label,
                                      build_character_model_mode, sow)
        lexicon.add_eos(self.int_eos_label, self.int_eos_word)
        lexicon.add_self_loops(self.int_phi)

        with tempfile.TemporaryDirectory() as directory:
            fst_filename = directory + '/L.fst'
            sym_filename = fst_filename + '.syms'
            pdf_filename = fst_filename + '.pdf'
            lexicon.write_fst(fst_filename)
            self.th.write_table(sym_filename)
            fst.draw(sym_filename, sym_filename, fst_filename, pdf_filename)

    def test_linear_linear_trie_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'linear', 'trie', eoc=eoc)

    def test_linear_linear_flat_write_fst(self):
        for sow in (None, self.int_sow):
            for eoc in (None, self.int_eoc):
                self._test_write_fst(Linear, 'linear', 'flat', sow, eoc)

    def test_linear_linear_copy_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'linear', 'copy', eoc=eoc)

    def test_linear_linear_linear_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'linear', 'linear', eoc=eoc)

    def test_linear_trie_trie_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'trie', 'trie', eoc=eoc)

    def test_linear_trie_flat_write_fst(self):
        for sow in (None, self.int_sow):
            for eoc in (None, self.int_eoc):
                self._test_write_fst(Linear, 'trie', 'flat', sow, eoc)

    def test_linear_trie_copy_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'trie', 'copy', eoc=eoc)

    def test_linear_trie_linear_write_fst(self):
        for eoc in (None, self.int_eoc):
            self._test_write_fst(Linear, 'trie', 'linear', eoc=eoc)
