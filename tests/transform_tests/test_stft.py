import unittest
from nt.io.audioread import audioread
import numpy as np
from scipy import signal

from nt.transform.module_stft import _samples_to_stft_frames
from nt.transform.module_stft import _stft_frames_to_samples
from nt.transform.module_stft import stft
from nt.transform.module_stft import istft
from nt.transform.module_stft import _biorthogonal_window_for
from nt.transform.module_stft import _biorthogonal_window_vec
from nt.transform.module_stft import stft_to_spectrogram
from nt.transform.module_stft import spectrogram_to_energy_per_frame
from pymatbridge import Matlab


class TestSTFTMethods(unittest.TestCase):
    def test_samples_to_stft_frames(self):
        size = 1024
        shift = 256

        self.assertEqual(_samples_to_stft_frames(1023, size, shift), 1)
        self.assertEqual(_samples_to_stft_frames(1024, size, shift), 1)
        self.assertEqual(_samples_to_stft_frames(1025, size, shift), 2)
        self.assertEqual(_samples_to_stft_frames(1024 + 256, size, shift), 2)
        self.assertEqual(_samples_to_stft_frames(1024 + 257, size, shift), 3)

    def test_stft_frames_to_samples(self):
        size = 1024
        shift = 256

        self.assertEqual(_stft_frames_to_samples(1, size, shift), 1024)
        self.assertEqual(_stft_frames_to_samples(2, size, shift), 1024 + 256)

    def test_restore_time_signal_from_stft_and_istft(self):
        path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        x = audioread(path)
        X = stft(x)
        spectrogram = stft_to_spectrogram(X)
        energy = spectrogram_to_energy_per_frame(spectrogram)

        self.assertTrue(np.allclose(x, istft(X, 1024, 256)[:len(x)]))
        self.assertEqual(X.shape, (186, 513))
        self.assertEqual(spectrogram.shape, (186, 513))

        self.assertGreaterEqual(spectrogram.all(), 0)  # spectrogram >= 0
        self.assertEqual(spectrogram.all().imag, 0)  # Im(spectrogram) == 0
        self.assertGreaterEqual(energy.all(), 0)  # energy >= 0
        self.assertEqual(energy.all().imag, 0)  # Im(energy) ?= 0
        self.assertEqual(energy.shape, (186,))

    def test_compare_both_biorthogonal_window_variants(self):
        window = signal.blackman(1024)
        shift = 256
        for_result = _biorthogonal_window_for(window, shift)
        vec_result = _biorthogonal_window_vec(window, shift)

        self.assertEqual(for_result.all(), vec_result.all())
        self.assertEqual(for_result.shape, (1024,))

    def compare_with_matlab(self):
        path = '/net/speechdb/timit/pcm/train/dr1/fcjf0/sa1.wav'
        y = audioread(path)
        Y_python = stft(y)

        # mlab = Matlab('nice -n 3 matlab -nodisplay -nosplash')
        mlab = Matlab()
        mlab.start()
        _ = mlab.run_code('run /net/home/ldrude/Projects/2015_python_matlab/matlab/startup.m')
        mlab.set_variable('y', y)
        mlab.run_code('Y = transform.stft(y(:), 1024, 256, @blackman);')
        # mlab.run_code('Y(1:10) = 0;')
        Y_matlab = mlab.get_variable('Y').T

        # np.testing.assert_almost_equal(Y_matlab, Y_python)
        # self.assertTrue(np.allclose(Y_matlab, Y_python))
        self.assertEqual(Y_matlab.all(), Y_python.all())
