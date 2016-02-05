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
from nt.speech_recognition import arpa
from nt.transcription_handling import lexicon
from nt.transcription_handling.transcription_handler import TranscriptionHandler


class TestDecoder(unittest.TestCase):
    def setUp(self):

        trans_handler_path = data_dir('speech_recognition', 'trans_handler')
        with open(trans_handler_path, 'rb') as fid:
            self.trans_handler = pickle.load(fid)

        self.tmpdir = tempfile.TemporaryDirectory()
        print(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # @unittest.skip("")
    def test_ground_truth(self):

        working_dir = self.tmpdir.name

        lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
        arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                           lm_path_uni)

        self.decoder = Decoder(self.trans_handler, working_dir,
                               lm_file=lm_path_uni)

        self.decoder.create_graphs()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        symbol_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        trans_hat = -100 * np.ones(
                (len(symbol_seq), 1, len(self.trans_handler.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0,
                      self.trans_handler.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_ground_truth_with_sil(self):

        space = "<SPACE>"
        self.trans_handler.label_handler.add_label(space)

        working_dir = self.tmpdir.name

        lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
        arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                           lm_path_uni)

        space = "<SPACE>"
        self.decoder = Decoder(self.trans_handler, working_dir,
                               lm_file=lm_path_uni, sil=space)

        self.decoder.create_graphs()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        symbol_seq = "T_HI__S__  ___SSHOOO_ULD__  BE___RECO_GNIIZ_ED____"

        trans_hat = -100 * np.ones(
                (len(symbol_seq), 1, len(self.trans_handler.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            if sym == " ":
                sym = space
            trans_hat[idx, 0,
                      self.trans_handler.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])

        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_only_lex(self):

        working_dir = self.tmpdir.name
        lm_file = data_dir('speech_recognition', 'tcb05cnp')
        lexicon_file = os.path.join(working_dir, "lexicon.txt")
        arpa.create_lexicon(lm_file, lexicon_file)

        self.decoder = Decoder(self.trans_handler, self.tmpdir.name)

        self.decoder.create_graphs()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        symbol_seq = "T_HI__S__SSHOOO_ULDBE___RECO_GNIIZ_ED____"

        trans_hat = -100 * np.ones(
                (len(symbol_seq), 1, len(self.trans_handler.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0,
                      self.trans_handler.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        self.decoder = Decoder(self.trans_handler, self.tmpdir.name)

        nn = BLSTMModel(self.trans_handler.label_handler, lstm_cells=256,
                        fbank_filters=80)
        load_hdf5(data_dir('speech_recognition', '1.nnet'), nn)

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

        self.decoder.create_graphs()
        self.decoder.decode_graph = self.decoder.ctc_map_fst
        batch = dp_test.test_run()
        net_out = nn._propagate(nn.data_to_variable(batch['x']))
        utt_id = "TEST_UTT_1"
        net_out_list = [net_out.num, ]
        self.decoder.create_lattices(net_out_list, [utt_id, ])
        sym_decode_int, word_decode_int = \
            self.decoder.decode(lm_scale=1, out_type='ints')

        word_decode = \
            self.decoder.trans_handler.labels_int2sym(
                word_decode_int["TEST_UTT_1"])

        argmax_ctc = argmax_ctc_decode(
                net_out.num[:, 0, :],
                self.trans_handler.label_handler)
        print(word_decode)
        print(argmax_ctc)

        self.assertEqual(argmax_ctc, word_decode)

    # @unittest.skip("")
    def test_one_word_grammar(self):

        working_dir = self.tmpdir.name

        word = "TEST"
        utt_id = "TEST_UTT_1"
        utt_length = len(word)

        lm_file = data_dir('speech_recognition', "arpa_one_word")

        self.decoder = Decoder(self.trans_handler, working_dir,
                               lm_file=lm_file)

        self.decoder.create_graphs()

        trans_hat = np.zeros((utt_length, 1,
                              len(self.trans_handler.label_handler)))
        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])

        self.assertEqual(word, word_decode[utt_id])

    # @unittest.skip("")
    def test_two_word_grammar(self):
        # fails since missing in trans_handler.lexicon

        working_dir = self.tmpdir.name

        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_id = "TEST_UTT_1"
        utt_length = len(word1)

        trans_hat = np.zeros((utt_length, 1,
                              len(self.trans_handler.label_handler)))
        for idx in range(utt_length):
            sym = word1[idx]
            # represents reward since net provides some kind of positive
            # loglikelihoods (with some offset)
            trans_hat[
                idx, 0, self.trans_handler.label_handler.label_to_int[sym]
            ] = 1 / utt_length

        trans_hat = Variable(trans_hat)

        lm_file = data_dir('speech_recognition', "arpa_two_words_uni")
        lexicon_file = os.path.join(working_dir, "lexicon.txt")
        arpa.create_lexicon(lm_file, lexicon_file)

        self.decoder = Decoder(self.trans_handler, working_dir,
                               lm_file=lm_file)

        self.decoder.create_graphs()

        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])

        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1.1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(word2, word_decode[utt_id])

        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=0.9, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])
        self.assertEqual(word1, word_decode[utt_id])

    # @unittest.skip("")
    def test_trigram_grammar(self):

        working_dir = self.tmpdir.name

        utt_id = "TEST_UTT_1"
        utt = "SHE SEES"
        symbol_seq = "SHE___SE_ES"
        trans_hat = np.zeros((len(symbol_seq), 1,
                              len(self.trans_handler.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[
                idx, 0, self.trans_handler.label_handler.label_to_int[sym]] = 1
        trans_hat = Variable(trans_hat)

        lm_file = data_dir('speech_recognition', "arpa_three_words_tri")
        lexicon_file = os.path.join(working_dir, "lexicon.txt")
        arpa.create_lexicon(lm_file, lexicon_file)

        self.decoder = Decoder(self.trans_handler, working_dir,
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

        working_dir = self.tmpdir.name

        lm_path_uni = os.path.join(working_dir, 'tcb05cnp')
        arpa.write_unigram(data_dir('speech_recognition', 'tcb05cnp'),
                           lm_path_uni)

        word1 = "WORKS"
        word2 = "KORKS"

        lex_file = os.path.join(working_dir, 'lexicon.txt')
        with open(lex_file, 'w') as fid:
            fid.write(word1)
            for letter in word1:
                fid.write(" " + letter)

        lex = lexicon.from_txt_file(lex_file)
        trans_handler = TranscriptionHandler(lex)

        self.decoder = Decoder(trans_handler, working_dir,
                               lm_file=lm_path_uni)

        self.decoder.create_graphs()

        utt_id = "TEST_UTT_1"
        utt_length = len(word1)

        trans_hat = np.zeros((utt_length, 1, len(trans_handler.label_handler)))
        for idx in range(len(word1)):
            sym1 = word1[idx]
            sym2 = word2[idx]
            trans_hat[idx, 0, trans_handler.label_handler.label_to_int[sym1]]\
                += 5
            trans_hat[idx, 0, trans_handler.label_handler.label_to_int[sym2]]\
                += 10

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = \
            self.decoder.decode(lm_scale=1, out_type='string')
        print(sym_decode[utt_id])
        print(word_decode[utt_id])

        self.assertEqual(word1, word_decode[utt_id])
