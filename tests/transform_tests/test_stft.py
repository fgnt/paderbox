import unittest

import numpy as np
from scipy import signal

import nt.testing as tc
from nt.io.audioread import audioread
from nt.io.data_dir import testing as testing_dir
from nt.transform.module_stft import _biorthogonal_window
from nt.transform.module_stft import _biorthogonal_window_loopy
from nt.transform.module_stft import _biorthogonal_window_brute_force
from nt.transform.module_stft import _biorthogonal_window_fastest
from nt.transform.module_stft import _samples_to_stft_frames
from nt.transform.module_stft import _stft_frames_to_samples
from nt.transform.module_stft import get_stft_center_frequencies
from nt.transform.module_stft import istft
from nt.transform.module_stft import spectrogram_to_energy_per_frame
from nt.transform.module_stft import stft
from nt.transform.module_stft import stft_single_channel
from nt.transform.module_stft import stft_to_spectrogram
from nt.utils.matlab import matlab_test, Mlab


class TestSTFTMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        path = testing_dir('timit', 'data', 'sample_1.wav')
        self.x = audioread(path)

    def test_samples_to_stft_frames(self):
        size = 1024
        shift = 256

        tc.assert_equal(_samples_to_stft_frames(1023, size, shift), 1)
        tc.assert_equal(_samples_to_stft_frames(1024, size, shift), 1)
        tc.assert_equal(_samples_to_stft_frames(1025, size, shift), 2)
        tc.assert_equal(_samples_to_stft_frames(1024 + 256, size, shift), 2)
        tc.assert_equal(_samples_to_stft_frames(1024 + 257, size, shift), 3)

    def test_stft_frames_to_samples(self):
        size = 1024
        shift = 256

        tc.assert_equal(_stft_frames_to_samples(1, size, shift), 1024)
        tc.assert_equal(_stft_frames_to_samples(2, size, shift), 1024 + 256)

    def test_restore_time_signal_from_stft_and_istft(self):
        x = self.x
        X = stft(x)

        tc.assert_almost_equal(x, istft(X, 1024, 256)[:len(x)])
        tc.assert_equal(X.shape, (186, 513))

    def test_restore_time_signal_from_stft_and_istft_kaldi_params(self):
        x = self.x
        X = stft(x, size=400, shift=160)

        tc.assert_almost_equal(x, istft(X, 400, 160)[:len(x)])
        tc.assert_equal(X.shape, (294, 201))

    def test_spectrogram_and_energy(self):
        x = self.x
        X = stft(x)
        spectrogram = stft_to_spectrogram(X)
        energy = spectrogram_to_energy_per_frame(spectrogram)

        tc.assert_equal(X.shape, (186, 513))

        tc.assert_equal(spectrogram.shape, (186, 513))
        tc.assert_isreal(spectrogram)
        tc.assert_array_greater_equal(spectrogram, 0)

        tc.assert_equal(energy.shape, (186,))
        tc.assert_isreal(energy)
        tc.assert_array_greater_equal(energy, 0)

    def test_compare_both_biorthogonal_window_variants(self):
        window = signal.blackman(1024)
        shift = 256

        for_result = _biorthogonal_window_loopy(window, shift)
        vec_result = _biorthogonal_window(window, shift)
        brute_force_result = _biorthogonal_window_brute_force(window, shift)

        tc.assert_equal(for_result, vec_result)
        tc.assert_allclose(for_result, brute_force_result)
        tc.assert_equal(for_result.shape, (1024,))

    def test_biorthogonal_window_inverts_analysis_window(self):
        from nt.utils.numpy_utils import roll_zeropad

        def inf_shift_add(analysis_window, shift):
            influence_width = ((len(analysis_window) - 1) // shift)
            influence_width *= 2  # be sure that it is high enough

            res = np.zeros_like(analysis_window)
            for i in range(-influence_width, influence_width + 1):
                res += roll_zeropad(analysis_window, shift * i)
            return res

        window = signal.blackman(1024)
        shift = 256

        synthesis_window = _biorthogonal_window_brute_force(window, shift)

        s = inf_shift_add(window * synthesis_window, shift)
        tc.assert_allclose(s, 1)

    def test_biorthogonal_window_inverts_analysis_window_kaldi_parameter(self):
        from nt.utils.numpy_utils import roll_zeropad

        def inf_shift_add(analysis_window, shift):
            influence_width = ((len(analysis_window) - 1) // shift)
            influence_width *= 2  # be sure that it is high enough

            res = np.zeros_like(analysis_window)
            for i in range(-influence_width, influence_width + 1):
                res += roll_zeropad(analysis_window, shift * i)
            return res

        window = signal.blackman(400)
        shift = 160

        synthesis_window = _biorthogonal_window_brute_force(window, shift)

        s = inf_shift_add(window * synthesis_window, shift)
        tc.assert_allclose(s, 1)

    def test_biorthogonal_window_fastest_is_fastest(self):
        from nt.utils.timer import TimerAccumulateDict
        timer = TimerAccumulateDict()

        window = signal.blackman(1024)
        shift = 256

        with timer['loopy']:
            for_result = _biorthogonal_window_loopy(window, shift)
        with timer['normal']:
            vec_result = _biorthogonal_window(window, shift)
        with timer['brute_force']:
            brute_force_result = _biorthogonal_window_brute_force(window, shift)
        with timer['fastest']:
            brute_force_result = _biorthogonal_window_fastest(window, shift)

        # brute_force is fastest
        # tc.assert_array_greater(timer.as_dict['fastest'] * ..., timer.as_dict['brute_force'])
        tc.assert_array_less(timer.as_dict['fastest'] * 10, timer.as_dict['normal'])
        tc.assert_array_less(timer.as_dict['fastest'] * 5, timer.as_dict['loopy'])

    def test_batch_mode(self):
        size = 1024
        shift = 256

        # Reference
        X = stft_single_channel(self.x)

        x1 = np.array([self.x, self.x])
        X1 = stft(x1)
        tc.assert_equal(X1.shape, (2, 186, 513))

        for d in np.ndindex(2):
            tc.assert_equal(X1[d, :, :].squeeze(), X)

        x11 = np.array([x1, x1])
        X11 = stft(x11)
        tc.assert_equal(X11.shape, (2, 2, 186, 513))
        for d, k in np.ndindex(2, 2):
            tc.assert_equal(X11[d, k, :, :].squeeze(), X)

        x2 = x1.transpose()
        X2 = stft(x2)
        tc.assert_equal(X2.shape, (186, 513, 2))
        for d in np.ndindex(2):
            tc.assert_equal(X2[:, :, d].squeeze(), X)

        x21 = np.array([x2, x2])
        X21 = stft(x21)
        tc.assert_equal(X21.shape, (2, 186, 513, 2))
        for d, k in np.ndindex(2, 2):
            tc.assert_equal(X21[d, :, :, k].squeeze(), X)

        x22 = x21.swapaxes(0, 1)
        X22 = stft(x22)
        tc.assert_equal(X22.shape, (186, 513, 2, 2))
        for d, k in np.ndindex(2, 2):
            tc.assert_equal(X22[:, :, d, k].squeeze(), X)

    def test_center_frequencies(self):
        tc.assert_allclose(get_stft_center_frequencies(size=1024, sample_rate=16000)[0], 0)

    @matlab_test
    def test_compare_with_matlab(self):
        y = self.x
        Y_python = stft(y)
        mlab = Mlab().process
        mlab.set_variable('y', y)
        mlab.run_code('Y = transform.stft(y(:), 1024, 256, @blackman);')
        Y_matlab = mlab.get_variable('Y').T
        tc.assert_almost_equal(Y_matlab, Y_python)
