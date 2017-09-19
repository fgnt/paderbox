import sys
import unittest
import os
import tempfile

import numpy as np

from nt.io.data_dir import testing as data_dir
from nt.speech_recognition.utils.utils import write_lattice_file, argmax_decode
from nt.TODO.kaldi.decoder import Decoder
from nt.transcription_handling.lexicon_handling import *
from nt.transcription_handling.transcription_handler import \
    LabelHandler


sys.path.append(str(data_dir / 'speech_recognition'))


class TestDecoder(unittest.TestCase):

    def write_net_out(self, labels, label_seq, cost_along_path=None):
        net_out = np.zeros((len(label_seq), 1, len(labels)))
        if cost_along_path:
            # net provides some kind of positive loglikelihoods
            # (with some offset)
            for idx in range(len(label_seq)):
                sym = label_seq[idx]
                if sym == "_":
                    sym = labels[0]
                if sym == " ":
                    sym = "<space>"
                net_out[idx, 0, labels.index(sym)] = \
                    -cost_along_path/len(label_seq)
        return net_out

    # @unittest.skip("")
    def test_ground_truth(self):

        lex = get_lexicon_from_arpa(
            str(data_dir / 'speech_recognition' / 'tcb05cnp'))
        labels = ["<blank>", ] + lex.tokens()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(labels, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(
                str(data_dir / 'speech_recognition' / 'tcb05cnp'),
                lm_path_uni)

            self.decoder = Decoder(labels, working_dir, lex,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_ground_truth_with_eow(self):

        space = "<space>"
        lex = get_lexicon_from_arpa(
            str(data_dir / 'speech_recognition' / 'tcb05cnp'))
        labels = ["<blank>", space] + lex.tokens()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__  ___SSHOOO_ULD__  BE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(labels, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(
                str(data_dir / 'speech_recognition' / 'tcb05cnp'),
                lm_path_uni)
            self.decoder = Decoder(labels, working_dir, lex,
                                   lm_file=lm_path_uni, silent_tokens=[space])
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_only_lex(self):

        lex = get_lexicon_from_arpa(
            str(data_dir / 'speech_recognition' / 'tcb05cnp'))
        labels = ["<blank>"] + lex.tokens()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(labels, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(labels, working_dir, lex)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        labels = ["<blank>", "a", "b"]

        net_out = np.random.randn(20, 1, len(labels))
        utt_id = "11111111"

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(labels, working_dir)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1)

        ref_decode, _ = argmax_decode(
            net_out[:, 0, :], transcription_handler=LabelHandler(labels))
        print(sym_decode)
        print(word_decode)
        print(ref_decode)
        self.assertEqual(ref_decode, word_decode[utt_id])

    # @unittest.skip("")
    def test_one_word_grammar(self):
        lm_file = str(data_dir / 'speech_recognition' / "arpa_one_word")
        lex = get_lexicon_from_arpa(lm_file)
        labels = ["<blank>"] + lex.tokens()

        word = "TEST"
        utt_id = "11111111"

        net_out = self.write_net_out(labels, word)
        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(labels, working_dir, lex, lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(word, word_decode[utt_id])

    # @unittest.skip("")
    def test_lm_scale(self):

        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_id = "11111111"

        lm_file = str(data_dir / 'speech_recognition' / "arpa_two_words_uni")
        lex = get_lexicon_from_arpa(lm_file)
        labels = ["<blank>"] + lex.tokens()

        net_out = self.write_net_out(labels, word1, -1)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(labels, working_dir, lex, lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode_ac, word_decode_ac = self.decoder.decode(
                search_graphs, lm_scale=0.9, out_type='string')
            sym_decode_lang, word_decode_lang = self.decoder.decode(
                search_graphs, lm_scale=1.1, out_type='string')

        print(sym_decode_ac[utt_id])
        print(word_decode_ac[utt_id])
        self.assertEqual(word1, word_decode_ac[utt_id])

        print(sym_decode_lang[utt_id])
        print(word_decode_lang[utt_id])
        self.assertEqual(word2, word_decode_lang[utt_id])

    # @unittest.skip("")
    def test_trigram_grammar(self):

        lm_file = str(data_dir / 'speech_recognition' / "arpa_three_words_tri")
        lex = get_lexicon_from_arpa(lm_file)
        labels = ["<blank>"] + lex.tokens()

        utt1_id = "11111111"
        utt1 = "SHE SEES"
        symbol_seq1 = "SHE___SE_ES__"
        net_out1 = self.write_net_out(labels, symbol_seq1, -10)
        utt2_id = "11111112"
        utt2 = "SHE SEES ME"
        symbol_seq2 = "SHE___SE_ES__MMEEE__"
        net_out2 = self.write_net_out(labels, symbol_seq2, -10)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(labels, working_dir, lex, lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt1_id: net_out1,
                                utt2_id: net_out2}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt1, word_decode[utt1_id])
        self.assertEqual(utt2, word_decode[utt2_id])

    # @unittest.skip("")
    def test_oov(self):
        with tempfile.TemporaryDirectory() as working_dir:
            word1 = "WORKS"
            word2 = "KORKS"

            lex_file = os.path.join(working_dir, 'lexicon.txt')
            with open(lex_file, 'w') as fid:
                fid.write(word1)
                for letter in word1:
                    fid.write(" " + letter)

            lex = get_lexicon_from_txt_file(lex_file)
            labels = ["<blank>"] + lex.tokens()

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(
                str(data_dir / 'speech_recognition' / 'tcb05cnp'),
                lm_path_uni)

            self.decoder = Decoder(labels, working_dir, lex,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()

            utt_id = "11111111"
            utt_length = len(word1)

            net_out = np.zeros((utt_length, 1, len(labels)))
            for idx in range(len(word1)):
                sym1 = word1[idx]
                sym2 = word2[idx]
                net_out[idx, 0, labels.index(sym1)] += 5
                net_out[idx, 0, labels.index(sym2)] += 10

            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = self.decoder.decode(
                search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)

        self.assertEqual(word1, word_decode[utt_id])

    def test_nbest(self):
        utt_id = "11111111"
        net_out = np.zeros((3, 1, 4))
        labels = ["<blk>", "a", "b", "c"]
        lex = {"abc": ["a", "b", "c"],
               "a": ["a"], }
        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(labels, working_dir, lex)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net1.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decodes_1, word_decodes_1 = self.decoder.decode(
                search_graphs, lm_scale=1, n=10, out_type='string')
            lattice_file = path.join(working_dir, "net2.lat")
            write_lattice_file({utt_id: net_out}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(
                lattice_file, determinize=False)
            sym_decodes_2, word_decodes_2 = self.decoder.decode(
                search_graphs, lm_scale=1, n=10, out_type='string')

        print(sym_decodes_1)
        print(word_decodes_1)
        print(sym_decodes_2)
        print(word_decodes_2)
        self.assertEqual(len(word_decodes_1), 4)
        self.assertEqual(len(word_decodes_2), 9)
        for decode in ["", "a", "a a", "abc"]:
            self.assertIn(decode, word_decodes_1.values())
            self.assertIn(decode, word_decodes_2.values())
