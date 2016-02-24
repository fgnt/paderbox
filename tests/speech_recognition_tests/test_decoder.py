import unittest
import numpy as np
import pickle
from nt.speech_recognition.decoder import Decoder
from chainer import Variable
from nt.nn import DataProvider
from nt.nn.data_fetchers.json_callback_fetcher import JsonCallbackFetcher
import sys
from nt.io.data_dir import testing as data_dir
from nt.io.data_dir import database_jsons as database_jsons_dir

sys.path.append(data_dir('speech_recognition'))
from model import BLSTMModel
from nt.utils.transcription_handling import argmax_ctc_decode
import os
import json
import tempfile
from chainer.serializers.hdf5 import load_hdf5
from nt.transcription_handling.lexicon_handling import *
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
                if sym == " ":
                    sym = "<space>"
                trans_hat[
                    idx, 0, trans_handler.label_handler.label_to_int[sym]]\
                    = -cost_along_path/len(label_seq)
        return Variable(trans_hat)

    def load_model(self):
        trans_handler_path = data_dir('speech_recognition', 'trans_handler')
        with open(trans_handler_path, 'rb') as fid:
            trans_handler = pickle.load(fid)

        #temporary workaround
        trans_handler.lexicon = {word: list(labels) for word, labels
                                 in trans_handler.lexicon.items()}

        nn = BLSTMModel(trans_handler.label_handler, lstm_cells=256,
                        fbank_filters=80)
        load_hdf5(data_dir('speech_recognition', 'best.nnet'), nn)
        return nn, trans_handler

    # @unittest.skip("")
    def test_ground_truth(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        trans_handler = TranscriptionHandler(lex)

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        trans_hat = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_ground_truth_with_sil(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        space = "<space>"
        trans_handler = TranscriptionHandler(lex, sil=space)

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        label_seq = "T_HI__S__  ___SSHOOO_ULD__  BE___RECO_GNIIZ_ED____"

        trans_hat = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:
            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)
            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()

            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_only_lex(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition', 'tcb05cnp'))
        trans_handler = TranscriptionHandler(lex)

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        label_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        trans_hat = self.write_net_out(trans_handler, label_seq, -1000)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir)
            self.decoder.create_graphs()
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        nn, trans_handler = self.load_model()

        trans_handler.sil = None  # temporary workaround

        json_path = database_jsons_dir('wsj.json')
        flist_test = 'test/flist/wave/official_si_dt_05'

        with open(json_path) as fid:
            json_data = json.load(fid)
        feature_fetcher_test = JsonCallbackFetcher(
            'x', json_data, flist_test, nn.transform_features,
            feature_channels=['observed/ch1'], nist_format=True)

        dp_test = DataProvider((feature_fetcher_test,),
                               batch_size=1,
                               shuffle_data=True)

        print(dp_test.data_info)
        batch = dp_test.test_run()

        net_out = nn._propagate(nn.data_to_variable(batch['x']))
        net_out_list = [net_out.num, ]

        utt_id = "TEST_UTT_1"

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(trans_handler, working_dir)
            self.decoder.create_graphs()
            self.decoder.decode_graph = self.decoder.ctc_map_fst
            self.decoder.create_lattices(net_out_list, [utt_id, ])
            sym_decode_int, word_decode_int = \
                self.decoder.decode(lm_scale=1, out_type='ints')
            word_decode = \
                self.decoder.trans_handler.ints2labels(
                    word_decode_int["TEST_UTT_1"])

        argmax_ctc = argmax_ctc_decode(net_out.num[:, 0, :],
                                       trans_handler.label_handler)
        print(word_decode)
        print(argmax_ctc)
        self.assertEqual(argmax_ctc, word_decode)

    # @unittest.skip("")
    def test_one_word_grammar(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition',
                                             'tcb05cnp'))
        trans_handler = TranscriptionHandler(lex)

        lm_file = data_dir('speech_recognition', "arpa_one_word")

        word = "TEST"
        utt_id = "TEST_UTT_1"

        trans_hat = self.write_net_out(trans_handler, word)

        with tempfile.TemporaryDirectory() as working_dir:
            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(word, word_decode[utt_id])

    # @unittest.skip("")
    def test_lm_scale(self):

        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_id = "TEST_UTT_1"

        lm_file = data_dir('speech_recognition', "arpa_two_words_uni")

        lex = get_lexicon_from_arpa(lm_file)
        trans_handler = TranscriptionHandler(lex)

        trans_hat = self.write_net_out(trans_handler, word1, -1)

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode_ac, word_decode_ac = \
                self.decoder.decode(lm_scale=0.9, out_type='string')
            sym_decode_lang, word_decode_lang = \
                self.decoder.decode(lm_scale=1.1, out_type='string')

        print(sym_decode_ac[utt_id])
        print(word_decode_ac[utt_id])
        self.assertEqual(word1, word_decode_ac[utt_id])

        print(sym_decode_lang[utt_id])
        print(word_decode_lang[utt_id])
        self.assertEqual(word2, word_decode_lang[utt_id])

    # @unittest.skip("")
    def test_trigram_grammar(self):

        lex = get_lexicon_from_arpa(data_dir('speech_recognition',
                                             'tcb05cnp'))
        trans_handler = TranscriptionHandler(lex)

        utt_id = "TEST_UTT_1"
        utt = "SHE SEES"
        symbol_seq = "SHE___SE_ES"
        trans_hat = self.write_net_out(trans_handler, symbol_seq, -10)
        lm_file = data_dir('speech_recognition', "arpa_three_words_tri")

        with tempfile.TemporaryDirectory() as working_dir:

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_file)
            self.decoder.create_graphs()
            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

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

            lex = get_lexicon_from_txt(lex_file)
            trans_handler = TranscriptionHandler(lex)

            lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
            arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                               lm_path_uni)

            self.decoder = Decoder(trans_handler, working_dir,
                                   lm_file=lm_path_uni)
            self.decoder.create_graphs()

            utt_id = "TEST_UTT_1"
            utt_length = len(word1)

            trans_hat = np.zeros((utt_length, 1,
                                  len(trans_handler.label_handler)))
            for idx in range(len(word1)):
                sym1 = word1[idx]
                sym2 = word2[idx]
                trans_hat[idx, 0,
                          trans_handler.label_handler.label_to_int[sym1]]\
                    += 5
                trans_hat[idx, 0,
                          trans_handler.label_handler.label_to_int[sym2]]\
                    += 10
            trans_hat = Variable(trans_hat)

            self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
            sym_decode, word_decode = \
                self.decoder.decode(lm_scale=1, out_type='string')

        print(sym_decode[utt_id])
        print(word_decode[utt_id])

        self.assertEqual(word1, word_decode[utt_id])
