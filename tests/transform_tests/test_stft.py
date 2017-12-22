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
from nt.transform.module_stft import stft_to_spectrogram
from nt.transform.module_stft import stft_with_kaldi_dimensions
from nt.utils.matlab import Mlab
from numpy.fft import rfft
import numpy


def stft_single_channel(time_signal, size=1024, shift=256,
                        window=signal.blackman,
                        fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a single channel time
    signal. It is able to add additional zeros for fade-in and fade out and
    should yield an STFT signal which allows perfect reconstruction.

    Up to now, only a single channel time signal is possible.

    :param time_signal: Single channel time signal.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    assert len(time_signal.shape) == 1

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        time_signal = numpy.pad(time_signal, size - shift, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(len(time_signal), size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    time_signal = numpy.pad(time_signal,
                            (0, samples - len(time_signal)), mode='constant')

    # The range object contains the sample index
    # of the beginning of each frame.
    range_object = range(0, len(time_signal) - size + shift, shift)

    if window_length is None:
        window = window(size+1)[:-1]
    else:
        window = window(size+1)[:-1]
        window = numpy.pad(window, (0, size - window_length), mode='constant')
    windowed = numpy.array([(window * time_signal[i:i + size])
                            for i in range_object])
    return rfft(windowed)


class TestSTFTMethods(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        path = testing_dir / 'timit' / 'data' / 'sample_1.wav'
        self.x = audioread(path)[0]

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
        from nt.utils.timer import TimerDict
        timer = TimerDict()

        window = signal.blackman(1024)
        shift = 256

        with timer['loopy']:
            for_result = _biorthogonal_window_loopy(window, shift)
        with timer['normal']:
            vec_result = _biorthogonal_window(window, shift)
        with timer['brute_force']:
            brute_force_result = _biorthogonal_window_brute_force(
                window, shift)
        with timer['fastest']:
            brute_force_result = _biorthogonal_window_fastest(window, shift)

        # brute_force is fastest
        # tc.assert_array_greater(timer.as_dict['fastest'] * ..., timer.as_dict['brute_force'])
        tc.assert_array_less(
            timer.as_dict['fastest'] * 5,
            timer.as_dict['normal'])
        tc.assert_array_less(
            timer.as_dict['fastest'] * 2,
            timer.as_dict['loopy'])

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
        X2 = stft(x2, axis=0)
        tc.assert_equal(X2.shape, (186, 513, 2))
        for d in np.ndindex(2):
            tc.assert_equal(X2[:, :, d].squeeze(), X)

        x21 = np.array([x2, x2])
        X21 = stft(x21, axis=1)
        tc.assert_equal(X21.shape, (2, 186, 513, 2))
        for d, k in np.ndindex(2, 2):
            tc.assert_equal(X21[d, :, :, k].squeeze(), X)

        x22 = x21.swapaxes(0, 1)
        X22 = stft(x22, axis=0)
        tc.assert_equal(X22.shape, (186, 513, 2, 2))
        for d, k in np.ndindex(2, 2):
            tc.assert_equal(X22[:, :, d, k].squeeze(), X)

    def test_window_length(self):
        X = stft(self.x, 512, 160, window_length=400)
        x_hat = istft(X, 512, 160, window_length=400)

        X_ref = istft(stft(self.x, 400, 160), 400, 160)
        tc.assert_equal(X.shape, (294, 257))

        tc.assert_allclose(X_ref, x_hat, rtol=1e-6, atol=1e-6)

    def test_center_frequencies(self):
        tc.assert_allclose(get_stft_center_frequencies(size=1024, sample_rate=16000)[0], 0)

    @tc.attr.matlab
    def test_compare_with_matlab(self):
        y = self.x
        Y_python = stft(y, symmetric_window=True)
        mlab = Mlab().process
        mlab.set_variable('y', y)
        mlab.run_code('Y = transform.stft(y(:), 1024, 256, @blackman);')
        Y_matlab = mlab.get_variable('Y').T
        tc.assert_almost_equal(Y_matlab, Y_python)
