import pickle
import sys
import unittest

import numpy as np
from chainer import Variable

from nt.io.data_dir import testing as data_dir
from nt.speech_recognition.nhpylm_decoder import NHPYLMDecoder

sys.path.append(data_dir('speech_recognition'))
from model import BLSTMModel
import tempfile
from chainer.serializers.hdf5 import load_hdf5
from nt.transcription_handling.transcription_handler import TranscriptionHandler


class TestDecoder(unittest.TestCase):
    def write_net_out(self, trans_handler, label_seq, cost_along_path=None):
        trans_hat = np.zeros(
            (len(label_seq), 1, len(trans_handler.label_handler)))
        if cost_along_path:
            # net provides some kind of positive loglikelihoods
            # (with some offset)
            for idx in range(len(label_seq)):
                sym = label_seq[idx]
                if sym == "_":
                    sym = trans_handler.blank
                trans_hat[
                    idx, 0, trans_handler.label_handler.label_to_int[sym]] \
                    = -cost_along_path / len(label_seq)
        return Variable(trans_hat)

    def load_model(self):
        trans_handler_path = data_dir('speech_recognition', 'trans_handler')
        with open(trans_handler_path, 'rb') as fid:
            trans_handler = pickle.load(fid)

        # temporary workaround
        trans_handler.lexicon = {word: list(labels) for word, labels
                                 in trans_handler.lexicon.items()}

        nn = BLSTMModel(trans_handler.label_handler, lstm_cells=256,
                        fbank_filters=80)
        load_hdf5(data_dir('speech_recognition', 'best.nnet'), nn)
        return nn, trans_handler

    # @unittest.skip("")
    def test_simple(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "AA BA AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A____AA__CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=True,
                                       phi_penalty=1,
                                       sow_penalty=1)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            self.decoder.draw_search_graphs()
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_simple_excl_lex(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "<AABA> AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A____AA__CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=True,
                                       phi_penalty=1,
                                       sow_penalty=1,
                                       exclusive_lex=True)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            self.decoder.draw_search_graphs()
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_new_word(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "<A> AB AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A_____CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=False)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_new_word_excl_lex(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "<A> AB AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A_____CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=True, exclusive_lex=True)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_new_word_no_char(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "AA BC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A_____CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=True, disable_char_model=True)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_new_word_with_space(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "<A> AB AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_ AAA____B____ _A_____CCCC "

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon,
                                                 sil=' ')

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler)
            self.decoder.create_graphs(debug=True)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_lexicon(self):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'label_handler.pkl'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt = "<A> AB AC"
        utt_id = "TEST_UTT_1"
        label_seq = "AA_AAAA____B_A_____CCCC"

        trans_handler_net = TranscriptionHandler(trans_handler.lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G.fst')
            lexicon = trans_handler.lexicon.copy()
            lexicon.pop('AA')
            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler,
                                         lexicon=lexicon)
            self.decoder.create_graphs(debug=True)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def _test_big_grammar(self, label_seq, utt="THIS SHOULD BE RECOGNIZED",
                          exclusive_lex=False):

        with open(
                data_dir(
                    'speech_recognition', 'NHPYLM', 'G_2_1.fst.label_handler'),
                'rb') as fid:
            trans_handler = pickle.load(fid)

        utt_id = "TEST_UTT_1"

        lexicon = dict(
            THIS=list('THIS'),
            SHOULD=list('SHOULD'),
            BE=list('BE'),
            RECOGNIZED=list('RECOGNIZED')
        )

        trans_handler_net = TranscriptionHandler(lexicon)

        trans_hat = self.write_net_out(trans_handler_net, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path = data_dir('speech_recognition', 'NHPYLM', 'G_2_1.fst')

            self.decoder = NHPYLMDecoder(trans_handler_net, working_dir,
                                         lm_path, trans_handler,
                                         lexicon=lexicon)
            self.decoder.create_graphs(debug=False,
                                       sow_penalty=0,
                                       exclusive_lex=exclusive_lex)
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    def test_big_grammar_1(self):
        self._test_big_grammar("T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____")

    def test_big_grammar_1_excl(self):
        self._test_big_grammar("T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____",
                               exclusive_lex=True)

    def test_big_grammar_no_blanks(self):
        self._test_big_grammar("THIS_SHOULDBERECOGNIZED")

    def test_big_grammar_all_blanks(self):
        self._test_big_grammar("_T_H_I_S_S_H_O_U_L_D_B_E_R_E_C_O_G_N_I_Z_E_D_")

    def test_big_grammar_new_word(self):
        self._test_big_grammar(
            "_T_H_I_S_S_H_O_U_L_D_B_E_E_N_R_E_C_O_G_N_I_Z_E_D_",
            utt='THIS SHOULD BE <EN> RECOGNIZED')

    def test_big_grammar_new_word_excl(self):
        self._test_big_grammar(
            "_T_H_I_S_S_H_O_U_L_D_B_E_E_N_R_E_C_O_G_N_I_Z_E_D_",
            utt='THIS SHOULD BE <EN> RECOGNIZED',
            exclusive_lex=True)
