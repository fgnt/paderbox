import unittest
from nt.evaluation.pesq import pesq
from scipy.io import wavfile
import wave, struct
from nt.io.audioread import audioread




class TestPESQ(unittest.TestCase):

    def setUp(self):
        self.ref = 'data/speech.wav'
        self.deg = 'data/speech_bab_0dB.wav'
        # rate, refer = wavfile.read('data/speech.wav')
        # waveFile = wave.open('data/speech.wav', 'r')
        # length = waveFile.getnframes()
        # for i in range(0,length):
        #     waveData = waveFile.readframes(1)
        #     data = struct.unpack("<h", waveData)
        #
        # self.rate = rate
        # print('this is original data')
        # print(refer)
        # self.refer = refer
        self.refer = audioread(self.ref)
        self.rate = 16000

    def test_wb_scores(self):
        scores = pesq(self.ref, self.deg, 'wb', self.rate)
        self.assertEqual(scores, (1.083, 0))
        print(scores)

    def test_nb_scores(self):
        scores = pesq(self.ref, self.deg, 'nb', self.rate)
        self.assertEqual(scores, (1.607, 1.969))
        print(scores)

r = TestPESQ()
r.setUp()

print('here goes the nb')
r.test_nb_scores()
#print('here goes the wb')
#r.test_wb_scores()