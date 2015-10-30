import unittest
import numpy as np
import pickle
from nt.speech_recognition.decoder import Decoder
from chainer import Variable
from nt.nn import NeuralNetwork, DataProvider
from nt.nn.data_fetchers.json_callback_fetcher import JsonCallbackFetcher,\
    to_fbank, stack_to_batch, segment_data
from nt.nn.data_fetchers.chime_transcription_data_fetcher \
    import ChimeTranscriptionDataFetcher
from nt.utils.transcription_handling import argmax_ctc_decode
import copy
import json
from chainer import functions as func

def transform_features(data_list):
    number_of_filters = 80
    data = to_fbank(data_list, number_of_filters=number_of_filters)
    data = segment_data(data, segment_length=11, segment_step=3)
    return stack_to_batch(data, dtype=np.float32)

class TestDecoder(unittest.TestCase):
    def setUp(self):
        nn_path = "?"
        label_handler_path = "?"

        # NN validation

        self.nn = NeuralNetwork().load(nn_path)

        with open(label_handler_path, 'rb') as label_handler_fid:
            self.label_handler = pickle.load(label_handler_fid)
            label_handler_fid.close()

        json_path = '/net/storage/jheymann/speech_db/reverb/reverb.json'
        flist_test = 'test/flists/wav/si_et_1'

        with open(json_path) as fid:
            json_data = json.load(fid)
        feature_fetcher_test = \
            JsonCallbackFetcher('fbank', json_data, flist_test, transform_features,
                                feature_channels=['observed/CH1'])

        trans_fetcher_test = ChimeTranscriptionDataFetcher('trans', json_data,
                                                           flist_test)

        # label_handler_test = trans_fetcher_test.label_handler
        self.dp_test = DataProvider((feature_fetcher_test, trans_fetcher_test),
                               batch_size=1,
                               shuffle_data=True)
        self.dp_test.print_data_info()

        self.dp_test.__iter__()

        self.decoder = Decoder(self.label_handler, grammar_type="unigram")

    def tearDown(self):
        self.dp_test.shutdown()

    @unittest.skip("")
    def test_ground_truth(self):
        utt = "THIS SHOULD BE RECOGNIZED"
        symbol_seq = "T_HI__S __SSHOOO_ULD BE_ __RECO_GNIIZ_ED____"

        trans_hat = -100*np.ones((len(symbol_seq), 1, len(self.label_handler)))
        for idx in range(len(symbol_seq)):
            sym = symbol_seq[idx]
            if sym == "_":
                sym = "BLANK"
            trans_hat[idx, 0, self.label_handler.label_to_int[sym]] = 0

        trans_hat = Variable(trans_hat)
        decodes = self.decoder.run_decode(trans_hat,
                                          graphs=["B", "BDL", "BDLG"])
        for graph in decodes:
            self.assertEqual(utt, decodes[graph]["decode"])

    @unittest.skip("")
    def test_compare_argmax_ctc(self):
        batch = self.dp_test.__next__()
        
        self.nn.set_inputs(**batch)
        self.forward_net(self.nn)
        trans_hat = self.nn.outputs.trans_hat
        decodes = self.decoder.run_decode(trans_hat, graphs=["B"])
        argmax_ctc = argmax_ctc_decode(trans_hat.data[:, 0, :],
                                       self.label_handler)
        print(argmax_ctc)

        for graph in decodes:
            self.assertEqual(argmax_ctc, decodes[graph]["decode"])

    # #@unittest.skip("")
    # def test_cmpr_UBDL_UBDLG_noGraphCosts(self):
    #     batch = self.dp_test.__next__()
    #
    #     self.nn.set_inputs(**batch)
    #     self.forward_net(self.nn)
    #     trans_hat = self.nn.outputs.trans_hat
    #
    #     decodes = self.decoder.run_decode(trans_hat, graphs=["BDL", "BDLG"],
    #                                       lm_scale=0)
    #
    #     self.assertEqual(decodes["BDLG"]["decode"], decodes["BDL"]["decode"])

    @unittest.skip("")
    def test_one_word_grammar(self):
        word = "SHOULD"
        neg_cost = 1
        utt_length = len(word)
        trans_hat = np.zeros((utt_length, 1, len(self.label_handler)))
        trans_hat = Variable(trans_hat)

        with open("test/arpa.tmp", 'w') as arpa_fid:
            arpa_fid.write("\\data\\\nngram 1=3\nngram 2=0\nngram 3=0\n"
                           "\n\\1-grams:\n"
                           "0\t<s>\t0\n"
                           "0\t</s>\t0\n"
                           "{0}\t{1}\t{2}\n".format(neg_cost, word, 0) +
                           "\n\\2-grams:\n"
                           "\n\\3-grams:\n"
                           "\end\\\n")

        self.decoder.arpa_path = "test/arpa.tmp"
        self.decoder.grammar_type = "trigram"

        decodes = self.decoder.run_decode(trans_hat, graphs=["BDLG"])

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
    def forward_net(nn, dropout_rate=0.):
        def drop(x):
            return func.dropout(x, dropout_rate)

        h_in_1 = func.clipped_relu(nn.layers.l_in_1_norm(nn.layers.l_in_1(drop(nn.inputs.fbank))))
        h_in_2 = func.clipped_relu(nn.layers.l_in_2_norm(nn.layers.l_in_2(drop(h_in_1))))

        act_fw = nn.layers.l_x_fw_norm(nn.layers.l_x_fw(drop(h_in_2)))
        act_bw = nn.layers.l_x_bw_norm(nn.layers.l_x_bw(drop(h_in_2)))
        lstm_fw, _, _ = nn.layers.l_lstm_fw(act_fw)
        lstm_bw, _, _ = nn.layers.l_lstm_bw(act_bw)
        blstm = lstm_fw + lstm_bw

        relu_1 = func.clipped_relu(nn.layers.l_out_norm_1(nn.layers.l_out_1(drop(blstm))))
        relu_2 = func.clipped_relu(nn.layers.l_out_norm_2(nn.layers.l_out_2(drop(relu_1))))

        nn.outputs.trans_hat = nn.layers.l_output(relu_2)
        # nn.outputs.ctc_loss = func.ctc(nn.outputs.trans_hat, nn.inputs.trans)
        nn.outputs.trans = nn.inputs.trans

        wer = 0
        per = 0
        # for b in range(nn.inputs.trans.num.shape[0]):
        #     ref = label_handler.int_arr_to_label_seq(nn.inputs.trans.num[0, :].tolist())
        #     dec = argmax_ctc_decode(nn.outputs.trans_hat.num[:, 0, :], label_handler)
        #     wer += editdistance.eval(ref.split(), dec.split()) / len(ref.split()) * 100
        #     per += editdistance.eval(list(ref), list(dec)) / len(list(ref)) * 100
        nn.outputs.wer = wer / nn.inputs.trans.num.shape[0]
        nn.outputs.per = per / nn.inputs.trans.num.shape[0]