import unittest
from nt.transform import stft
from nt.io.audioread import audioread
import numpy as np
from scipy import signal

class TestSTFTMethods(unittest.TestCase):
    def test_samples_to_stft_frames(self):
        size = 1024
        shift = 256
        self.assertEqual(stft._samples_to_stft_frames(1023, size, shift), 1)
        self.assertEqual(stft._samples_to_stft_frames(1024, size, shift), 1)
        self.assertEqual(stft._samples_to_stft_frames(1025, size, shift), 2)
        self.assertEqual(stft._samples_to_stft_frames(1024+256, size, shift), 2)
        self.assertEqual(stft._samples_to_stft_frames(1024+257, size, shift), 3)

    def test_stft_frames_to_samples(self):
        size = 1024
        shift = 256
        self.assertEqual(stft._stft_frames_to_samples(1, size, shift), 1024)
        self.assertEqual(stft._stft_frames_to_samples(2, size, shift), 1024+256)

    def test_restore_time_signal_from_stft_and_istft(self):
        path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        x = audioread(path)
        X = stft.stft(x)
        np.testing.assert_almost_equal(x, stft.istft(X, 1024, 256)[:len(x)])

    def compare_both_biorthogonal_window_variants(self):
        window = signal.blackman(1024)
        shift = 256
        for_result = stft._biorthogonal_window_for(window, shift)
        vec_result = stft._biorthogonal_window_vec(window, shift)
        np.testing.assert_almost_equal(for_result, vec_result)
