__author__ = 'walter'
import unittest
import tempfile
from nt.speech_recognition.lexicon import Trie
from nt.transcription_handling.transcription_handler import TranscriptionHandler
from nt.speech_recognition import fst

class TestLexicon(unittest.TestCase):

    def test_write_fst(self):
        words = ['Mars', 'man', 'Martian', 'Marsman']
        lexicon = {word: list(word) for word in words}

        special_symbols = {'blank': '<blank>', 'eps': '<eps>', 'phi': '<phi>', 'sow': '<sow>',
                           'eow': '</eow>', 'sos': '<sos>', 'eos': '</eos>'}
        transcription_handler = TranscriptionHandler(lexicon, **special_symbols)

        word_offset = len(transcription_handler.label_handler)
        int_lexicon = {transcription_handler.words2ints(word)[0] + word_offset:
                           transcription_handler.labels2ints(labels) for word, labels in lexicon.items()}
        int_eps = transcription_handler.labels2ints(special_symbols['eps'])[0]
        int_eos = transcription_handler.labels2ints(special_symbols['eow'])[0]
        int_label = [transcription_handler.labels2ints(label)[0] for label in transcription_handler.label_handler.labels]

        word_trie = Trie(int_eps, int_eos)
        for word in int_lexicon.items():
            word_trie.add_word(word)

        word_trie.build_character_model(int_label)

        with tempfile.TemporaryDirectory() as directory:
            fst_filename = directory + '/L.fst'
            sym_filename = fst_filename + '.syms'
            pdf_filename = fst_filename + '.pdf'
            word_trie.write_fst(fst_filename)
            transcription_handler.write_table(sym_filename)
            fst.draw(sym_filename, sym_filename, fst_filename, pdf_filename)