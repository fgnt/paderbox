import numpy as np
import scipy

from scipy import signal

import pylab as plt
import seaborn as sns
sns.set_palette("deep", desat=.6)
cmap = sns.diverging_palette(220, 20, n=7, as_cmap=True)

from numpy.fft import rfft, irfft


def stft(signal, size=1024, shift=256, window=signal.blackman, fading=True):
    """
    Calculates the short time Fourier transformation of a single channel time
    signal. It is able to add additional zeros for fade-in and fade out and
    should yield an STFT signal which allows perfect reconstruction.

    Up to now, only a single channel time signal is possible.

    :param signal: Single channel time signal.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    assert(len(signal.shape) == 1)

    # Pad with zeros to have enough samples for the window function to fade.
    if fading is True:
        signal = np.pad(signal, size-shift, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(len(signal), size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    signal = np.pad(signal, (0, samples - len(signal)), mode='constant')

    # The range object contains the sample index of the beginning of each frame.
    range_object = range(0, len(signal) - size + shift, shift)

    window = window(size)

    return np.array([rfft(window*signal[i:i+size]) for i in range_object])


def _samples_to_stft_frames(samples, size, shift):
    return np.ceil((samples - size + shift) / shift)


def _stft_frames_to_samples(frames, size, shift):
    return frames * shift + size - shift


def _biorthogonal_window_for(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab impelementation in terms of variable names.

    The results are equal.
    """
    fft_size = len(analysis_window)
    assert(np.mod(fft_size, shift) == 0);
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts+1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window


def _biorthogonal_window_vec(analysis_window, shift):
    """
    This is a vectorized implementation of the window calculation. It is much
    slower than the variant using for loops.
    """
    fft_size = len(analysis_window)
    assert(np.mod(fft_size, shift) == 0)
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        sample_index = np.arange(0, number_of_shifts+1)
        analysis_index = synthesis_index + sample_index * shift
        analysis_index = analysis_index[analysis_index + 1 < fft_size]
        sum_of_squares[synthesis_index] = np.sum(analysis_window[analysis_index] ** 2)
    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size
    return synthesis_window


def istft(X, size=1024, shift=256, window=signal.blackman, fading=True):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    :param X: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Removes the additional padding, if done during STFT.
    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    assert(X.shape[1] == 1024 // 2 + 1)

    window = _biorthogonal_window_for(window(size), shift)

    # Why? Line created by Hai, Lukas does not know, why it exists.
    window = window * size

    x = scipy.zeros(X.shape[0] * shift + size - shift)

    for n, i in enumerate(range(0, len(x) - size + shift, shift)):
        x[i:i + size] += window * np.real(irfft(X[n]))

    # Compensate fade-in and fade-out
    if fading:
        x = x[size-shift:len(x)-(size-shift)]

    return x


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal. The output is guaranteed to be real.

    :param stft: Complex STFT signal with dimensions #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.
    """
    spectrogram = np.abs(stft_signal * np.conjugate(stft_signal))
    return spectrogram


def plot_spectrogram(spectrogram, limits=None):
    if limits is None:
        limits = (np.min(spectrogram), np.max(spectrogram))

    plt.imshow(np.clip(np.log10(spectrogram).T, limits[0], limits[1]),
               interpolation='none', origin='lower', cmap=cmap)
    plt.grid(False)
    plt.xlabel('Time frame')
    plt.ylabel('Frequency bin')
    cbar = plt.colorbar()
    cbar.set_label('Energy / dB')
    plt.show()


def plot_stft(stft_signal, limits=None):
    plot_spectrogram(stft_to_spectrogram(stft_signal), limits)
