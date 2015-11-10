import unittest
import numpy as np
import pickle
from nt.speech_recognition.decoder import Decoder
from chainer import Variable
from nt.nn import NeuralNetwork, DataProvider
from nt.nn.data_fetchers.json_callback_fetcher import JsonCallbackFetcher,\
    to_fbank, stack_to_batch, segment_data
from nt.utils.transcription_handling import argmax_ctc_decode
import os
import json
import tempfile

number_of_filters = 80
segment_length = 3
segment_step = 3
def transform_features(data_list):
    data = to_fbank(data_list, number_of_filters=number_of_filters)
    data = [np.log(d) for d in data]
    data = segment_data(data, segment_length=segment_length,
                        segment_step=segment_step)
    return stack_to_batch(data, dtype=np.float32)

class TestDecoder(unittest.TestCase):
    def setUp(self):
        data_dir = "/net/storage/jheymann/notebooks_nt/toolbox_examples/nn/data/2015-11-03-11-03-40_Full_WSJ_CTC_dropout_google_init_stack_3_skip_3"
        nn_path = os.path.join(data_dir, 'best.nnet')
        label_handler_path = os.path.join(data_dir, 'label_handler')

        self.nn = NeuralNetwork().load(nn_path)
        with open(label_handler_path, 'rb') as label_handler_fid:
            self.label_handler = pickle.load(label_handler_fid)
            label_handler_fid.close()

        self.graph_dir = tempfile.mkdtemp()
        # self.graph_dir = "test"
        print(self.graph_dir)

        json_path = '/net/ssd/jheymann/owncloud/python_nt/nt/database/wsj/wsj.json'
        flist_test = 'test/flist/wave/official_si_dt_05'

        with open(json_path) as fid:
            json_data = json.load(fid)
        self.feature_fetcher_test = JsonCallbackFetcher(
            'fbank', json_data, flist_test, transform_features,
            feature_channels=['observed/ch1'], nist_format=True)

        # label_handler_test = trans_fetcher_test.label_handler
        self.dp_test = DataProvider((self.feature_fetcher_test, ),
                               batch_size=1,
                               shuffle_data=True)
        self.dp_test.print_data_info()

        self.dp_test.__iter__()

    def tearDown(self):
        self.dp_test.shutdown()

    #@unittest.skip("")
    def test_ground_truth(self):

        self.decoder = Decoder(self.label_handler, self.graph_dir,
                      lm_file='/net/nas/ebbers/project_echo/wsj_system/lm/tcb05cnp', grammar_type="unigram")
        self.decoder.create_graphs()

        utt = "THIS SHOULD BE RECOGNIZED"
        utt_id = "TEST_UTT_1"
        symbol_seq = "T_HI__S __SSHOOO_ULD BE_ __RECO_GNIIZ_ED____"

        trans_hat = -100*np.ones((len(symbol_seq), 1, len(self.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices([trans_hat.num,], [utt_id,])
        sym_decode, word_decode = self.decoder.decode(lm_scale=1)
        self.assertEqual(utt, word_decode[utt_id])

    # @unittest.skip("")
    def test_compare_argmax_ctc(self):

        self.decoder = Decoder(self.label_handler, self.graph_dir,
                      lm_file='/net/nas/ebbers/project_echo/wsj_system/lm/tcb05cnp', grammar_type=None, use_lexicon=False)
        self.decoder.create_graphs()
        batch = self.dp_test.__next__()
        self.nn.set_inputs(**batch)
        self.forward(self.nn)
        utt_idx = self.dp_test.current_observation_indices[0]
        utt_id = self.feature_fetcher_test.utterance_ids[utt_idx]
        net_out_list = [self.nn.outputs.trans_hat.num,]
        self.decoder.create_lattices(net_out_list, [utt_id,])
        sym_decode, word_decode = self.decoder.decode(
            lm_scale=1, table_out="symbols")

        argmax_ctc = argmax_ctc_decode(
            self.nn.outputs.trans_hat.num[:, 0, :],
            self.label_handler)
        print(word_decode[utt_id])
        print(argmax_ctc)

        self.assertEqual(argmax_ctc, word_decode[utt_id])

    @unittest.skip("")
    def test_one_word_grammar(self):
        word = "SHOULD"
        neg_cost = 1
        utt_length = len(word)

        with open("test/arpa.tmp", 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=3\nngram 2=0\nngram 3=0\n"
                           "\n\\1-grams:\n"
                           "0\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(neg_cost, word, 0) +
                           "\n\\2-grams:\n"
                           "\n\\3-grams:\n"
                           "\end\\\n")

        self.decoder.lm_file = "test/arpa.tmp"
        self.decoder.grammar_type = "trigram"

        trans_hat = np.zeros((utt_length, 1, len(self.label_handler)))
        trans_hat = Variable(trans_hat)
        self.decoder.create_lattices(net_out_list, [utt_id,],
                                     use_lexicon=False, use_lm=False)
        decodes = self.decoder.decode(lm_scale=1)

        for graph in decodes:
            self.assertEqual(word, decodes[graph]["decode"])

    @unittest.skip("")
    def test_two_word_grammar(self):
        word1 = "ACOUSTIC"
        word2 = "LANGUAGE"
        utt_length = len(word1)
        trans_hat = np.zeros((utt_length, 1, len(self.label_handler)))
        for idx in range(utt_length):
            sym = word1[idx]
            # represents reward since net provides some kind of positive
            # loglikelihoods (with some offset)
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] =\
                1/utt_length

        trans_hat = Variable(trans_hat)

        neg_cost1 = 0
        neg_cost2 = 1/2.309
        # since scaling with -2.3 in arpa2fst ... maybe this is because of
        # bigram backoff fix=11
        with open("test/arpa.tmp", 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=4\nngram 2=0\nngram 3=0\n"
                           "\n\\1-grams:\n"
                           "0\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(neg_cost1, word1, 0) +
                           "{0}\t{1}\t{2}\n".format(neg_cost2, word2, 0) +
                           "\n\\2-grams:\n"
                           "\n\\3-grams:\n"
                           "\n\end\\\n")

        self.decoder.arpa_path = "test/arpa.tmp"
        self.decoder.grammar_type = "trigram"

        decodes = self.decoder.run_decode(trans_hat, graphs=["BDLG"],
                                          am_scale=0.9)
        for graph in decodes:
            self.assertEqual(word2, decodes[graph]["decode"])

        decodes = self.decoder.run_decode(trans_hat, graphs=["BDLG"],
                                          am_scale=1.1)

        for graph in decodes:
            self.assertEqual(word1, decodes[graph]["decode"])

    @unittest.skip("")
    def test_trigram_grammar(self):
        word1 = "HE"
        word2 = "SHE"
        word3 = "SEES"

        utt = "SHE SEES"
        symbol_seq = "SHE SE_ES"
        trans_hat = np.zeros((len(symbol_seq)+1, 1, len(self.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = 1
        trans_hat = Variable(trans_hat)

        with open("test/arpa.tmp", 'w') as arpa_fid:
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
                           "{0}\t{1}\t{2}\t{3}\n".format(0, "<s>", word1, word3) +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, "<s>", word2, word3) +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, word1, word3, "</s>") +
                           "{0}\t{1}\t{2}\t{3}\n".format(0, word2, word3, "</s>") +
                           "\n\end\\\n")

        self.decoder.arpa_path = "test/arpa.tmp"
        self.decoder.grammar_type = "trigram"

        decodes = self.decoder.run_decode(trans_hat, graphs=["BDLG"])

        for graph in decodes:
            self.assertEqual(utt, decodes[graph]["decode"])

    @staticmethod
    def forward(nn, dropout_rate=0., noise_var=0.):

        x = nn.layers.l_x_norm(nn.inputs.fbank)
        act_fw = nn.layers.l_x_fw(x)
        act_bw = nn.layers.l_x_bw(x)
        lstm_fw, _, _ = nn.layers.l_lstm_fw((act_fw))
        lstm_bw, _, _ = nn.layers.l_lstm_bw((act_bw))
        blstm = lstm_fw + lstm_bw

        act_fw_2 = nn.layers.l_x_fw_2(blstm)
        act_bw_2 = nn.layers.l_x_bw_2(blstm)
        lstm_fw_2, _, _ = nn.layers.l_lstm_fw_2(act_fw_2)
        lstm_bw_2, _, _ = nn.layers.l_lstm_bw_2(act_bw_2)
        blstm_2 = lstm_fw_2 + lstm_bw_2

        nn.outputs.trans_hat = nn.layers.l_output(blstm_2)