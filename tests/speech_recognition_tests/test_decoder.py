import unittest
import numpy as np
from nt.speech_recognition.decoder import Decoder
from nt.speech_recognition.utils.utils import write_lattice_file, argmax_decode
from chainer import Variable
import sys
from nt.io.data_dir import testing as data_dir

sys.path.append(data_dir('speech_recognition'))
from model import BLSTMModel
import os
import tempfile
from nt.transcription_handling.lexicon_handling import *
from nt.transcription_handling.transcription_handler import TranscriptionHandler


class TestDecoder(unittest.TestCase):

    def write_net_out(self, trans_handler, label_seq, cost_along_path=None):
        net_out = np.zeros((len(label_seq), 1, trans_handler.num_tokens))
        if cost_along_path:
            # net provides some kind of positive loglikelihoods
            # (with some offset)
            for idx in range(len(label_seq)):
                sym = label_seq[idx]
                if sym == "_":
                    sym = trans_handler.int_to_label[0]
                if sym == " ":
                    sym = "<space>"
                net_out[
                    idx, 0, trans_handler.label_to_int[sym]]\
                    = -cost_along_path/len(label_seq)
        return Variable(net_out)

    # @unittest.skip("")
    def test_ground_truth(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        mandatory_tokens = ["<blank>", ]
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=mandatory_tokens)

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_ground_truth_with_sil(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        space = "<space>"
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=["<blank>", ], eow=space)

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__  ___SSHOOO_ULD__  BE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)
            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_only_lex(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=["<blank>", ])

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "11111111"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        net_out = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        trans_handler = TranscriptionHandler(["a", "b"],
                                             mandatory_tokens=["<blank>", ])

        nn = BLSTMModel(10, 3, 2, 32)
        model_input = \
            np.random.uniform(0, 1, size=(20, 1, 10)).astype(np.float32)
        net_out = nn.propagate(nn.data_to_variable(model_input))

        utt_id = "11111111"

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(trans_handler, working_dir)
            self.decoder.create_graphs()
            self.decoder.decode_graph = self.decoder.ctc_map_fst
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1)

        ref_decode, _ = argmax_decode(
            net_out.num[:, 0, :], transcription_handler=trans_handler)
        print(sym_decode)
        print(word_decode)
        print(ref_decode)
        self.assertEqual(ref_decode, word_decode[utt_id])

    # @unittest.skip("")
    def test_one_word_grammar(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition',
                                             'tcb05cnp'))
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=["<blank>", ])

        lm_file = data_dir('speech_recognition', "arpa_one_word")

        word = "TEST"
        utt_id = "11111111"

        net_out = self.write_net_out(trans_handler, word)

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)
        self.assertEqual(word, word_decode[utt_id])

    # @unittest.skip("")
    def test_lm_scale(self):

        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_id = "11111111"

        lm_file = data_dir('speech_recognition', "arpa_two_words_uni")

        lex = get_lexicon_from_arpa(lm_file)
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=["<blank>", ])

        net_out = self.write_net_out(trans_handler, word1, -1)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode_ac, word_decode_ac = \
                self.decoder.decode(search_graphs, lm_scale=0.9, out_type='string')
            sym_decode_lang, word_decode_lang = \
                self.decoder.decode(search_graphs, lm_scale=1.1, out_type='string')

        print(sym_decode_ac[utt_id])
        print(word_decode_ac[utt_id])
        self.assertEqual(word1, word_decode_ac[utt_id])

        print(sym_decode_lang[utt_id])
        print(word_decode_lang[utt_id])
        self.assertEqual(word2, word_decode_lang[utt_id])

    # @unittest.skip("")
    def test_trigram_grammar(self):

        lm_file = data_dir('speech_recognition', "arpa_three_words_tri")
        lex = get_lexicon_from_arpa(data_dir('speech_recognition', "tcb05cnp"))
        trans_handler = TranscriptionHandler(
            lex, mandatory_tokens=["<blank>", ])

        utt1_id = "11111111"
        utt1 = "SHE SEES"
        symbol_seq1 = "SHE___SE_ES__"
        net_out1 = self.write_net_out(trans_handler, symbol_seq1, -10)
        utt2_id = "11111112"
        utt2 = "SHE SEES ME"
        symbol_seq2 = "SHE___SE_ES__MMEEE__"
        net_out2 = self.write_net_out(trans_handler, symbol_seq2, -10)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt1_id: net_out1.num,
                                utt2_id: net_out2.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

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
            trans_handler = TranscriptionHandler(
                lex, mandatory_tokens=["<blank>", ])

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()

            utt_id = "11111111"
            utt_length = len(word1)

            net_out = np.zeros((utt_length, 1, trans_handler.num_tokens))
            for idx in range(len(word1)):
                sym1 = word1[idx]
                sym2 = word2[idx]
                net_out[idx, 0,
                        trans_handler.label_to_int[sym1]]\
                    += 5
                net_out[idx, 0,
                        trans_handler.label_to_int[sym2]]\
                    += 10
            net_out = Variable(net_out)

            lattice_file = path.join(working_dir, "net.lat")
            write_lattice_file({utt_id: net_out.num}, lattice_file)
            search_graphs = self.decoder.create_search_graphs(lattice_file)
            sym_decode, word_decode = \
                self.decoder.decode(search_graphs, lm_scale=1, out_type='string')

        print(sym_decode)
        print(word_decode)

        self.assertEqual(word1, word_decode[utt_id])
