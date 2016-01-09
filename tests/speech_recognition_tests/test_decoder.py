data_dir = "/net/storage/python_unittest_data/speech_recognition"

import unittest
import numpy as np
import pickle
from nt.speech_recognition.decoder import Decoder
from chainer import Variable
from nt.nn import DataProvider
from nt.nn.data_fetchers.json_callback_fetcher import JsonCallbackFetcher
import sys

sys.path.append(data_dir)
from exp_models import BLSTMModel
from nt.utils.transcription_handling import argmax_ctc_decode
import os
import json
import tempfile
from chainer.serializers.hdf5 import load_hdf5


class TestDecoder(unittest.TestCase):
    def setUp(self):

        label_handler_path = os.path.join(data_dir, 'label_handler')
        with open(label_handler_path, 'rb') as label_handler_fid:
            self.label_handler = pickle.load(label_handler_fid)
            label_handler_fid.close()

        self.nn = BLSTMModel(self.label_handler)
        load_hdf5(os.path.join(data_dir, 'best.nnet'), self.nn)

        self.nn.load_mean_and_var(os.path.join(data_dir, 'mean_and_var_train'))

        json_path = '/net/storage/database_jsons/wsj.json'
        flist_test = 'test/flist/wave/official_si_dt_05'

        with open(json_path) as fid:
            json_data = json.load(fid)
        feature_fetcher_test = JsonCallbackFetcher(
                'fbank', json_data, flist_test, self.nn.transform_features,
                feature_channels=['observed/ch1'], nist_format=True)

        # label_handler_test = trans_fetcher_test.label_handler
        self.dp_test = DataProvider((feature_fetcher_test,),
                                    batch_size=1,
                                    shuffle_data=True)
        self.dp_test.print_data_info()

        self.tmpdir = tempfile.TemporaryDirectory()
        print(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # @unittest.skip("")
    def test_ground_truth(self):
        self.decoder = Decoder(self.label_handler, self.tmpdir.name,
                               lm_file=os.path.join(data_dir, 'tcb05cnp'),
                               grammar_type="unigram")
        self.decoder.create_graphs()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        symbol_seq = "T_HI__S __SSHOOO_ULD BE_ __RECO_GNIIZ_ED____"

        trans_hat = -100 * np.ones(
                (len(symbol_seq), 1, len(self.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = self.decoder.decode(lm_scale=1)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        self.decoder = Decoder(self.label_handler, self.tmpdir.name,
                               lm_file=os.path.join(data_dir, 'tcb05cnp'),
                               grammar_type=None, use_lexicon=False)
        self.decoder.create_graphs()
        batch = self.dp_test.test_run()
        net_out = self.nn._propagate(self.nn.data_to_variable(batch['x']))
        utt_id = "TEST_UTT_2"
        net_out_list = [net_out.num, ]
        self.decoder.create_lattices(net_out_list, [utt_id, ])
        sym_decode, word_decode = self.decoder.decode(lm_scale=1)

        argmax_ctc = argmax_ctc_decode(
                net_out.num[:, 0, :],
                self.label_handler)
        print(word_decode[utt_id])
        print(argmax_ctc)

        self.assertEqual(argmax_ctc, word_decode[utt_id])

    # @unittest.skip("")
    def test_one_word_grammar(self):

        word = "SHOULD"
        neg_cost = 1
        utt_id = "TEST_UTT_3"
        utt_length = len(word)

        lm_file = os.path.join(self.tmpdir.name, "arpa.tmp")
        with open(lm_file, 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=3\nngram 2=0\nngram 3=0\n"
                           "\n\\1-grams:\n"
                           "0\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(neg_cost, word, 0) +
                           "\n\\2-grams:\n"
                           "\n\\3-grams:\n"
                           "\end\\\n")

        self.decoder = Decoder(self.label_handler, self.tmpdir.name,
                               lm_file=lm_file, grammar_type="trigram")
        self.decoder.create_graphs()

        trans_hat = np.zeros((utt_length, 1, len(self.label_handler)))
        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = self.decoder.decode(lm_scale=1)

        self.assertEqual(word, word_decode[utt_id])

    # @unittest.skip("")
    def test_two_word_grammar(self):
        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_id = "TEST_UTT_4"
        utt_length = len(word1)

        trans_hat = np.zeros((utt_length, 1, len(self.label_handler)))
        for idx in range(utt_length):
            sym = word1[idx]
            # represents reward since net provides some kind of positive
            # loglikelihoods (with some offset)
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = \
                1 / utt_length

        trans_hat = Variable(trans_hat)

        neg_cost1 = 0
        neg_cost2 = 1 / np.log(10)
        # since scaling with -2.3 in arpa2fst ... maybe this is because of
        # bigram backoff fix=11
        lm_file = os.path.join(self.tmpdir.name, "arpa.tmp")
        with open(lm_file, 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=4\nngram 2=0\nngram 3=0\n"
                           "\n\\1-grams:\n"
                           "0\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(neg_cost1, word1, 0) +
                           "{0}\t{1}\t{2}\n".format(neg_cost2, word2, 0) +
                           "\n\\2-grams:\n"
                           "\n\\3-grams:\n"
                           "\n\end\\\n")

        self.decoder = Decoder(self.label_handler, self.tmpdir.name,
                               lm_file=lm_file, grammar_type="trigram")
        self.decoder.create_graphs()

        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])

        sym_decode, word_decode = self.decoder.decode(lm_scale=1.1)
        self.assertEqual(word2, word_decode[utt_id])

        sym_decode, word_decode = self.decoder.decode(lm_scale=0.9)
        self.assertEqual(word1, word_decode[utt_id])

    # @unittest.skip("")
    def test_trigram_grammar(self):
        word1 = "HE"
        word2 = "SHE"
        word3 = "SEES"

        utt_id = "TEST_UTT_5"
        utt = "SHE SEES"
        symbol_seq = "SHE SE_ES"
        trans_hat = np.zeros((len(symbol_seq), 1, len(self.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = 1
        trans_hat = Variable(trans_hat)

        lm_file = os.path.join(self.tmpdir.name, "arpa.tmp")
        with open(lm_file, 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=5\nngram 2=2\nngram 3=4\n"
                           "\n\\1-grams:\n"
                           "1\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(-2, word1, 0) +
                           "{0}\t{1}\t{2}\n".format(0, word2, 0) +
                           "{0}\t{1}\t{2}\n".format(-2, word3, 0) +
                           "\n\\2-grams:\n"
                           "{0}\t{1}\t{2}\t{3}\n".format(-1, word1, word3, 0) +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, word2, word3, 0) +
                           "\n\\3-grams:\n"
                           "{0}\t{1}\t{2}\t{3}\n".format(0, "<s>", word1,
                                                         word3) +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, "<s>", word2,
                                                         word3) +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, word1, word3,
                                                         "</s>") +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, word2, word3,
                                                         "</s>") +
                           "\n\end\\\n")

        self.decoder = Decoder(self.label_handler, self.tmpdir.name,
                               lm_file=lm_file, grammar_type="trigram")
        self.decoder.create_graphs()

        self.decoder.create_lattices([trans_hat.num, ], [utt_id, ])
        sym_decode, word_decode = self.decoder.decode(lm_scale=1)
        self.assertEqual(utt, word_decode[utt_id])
