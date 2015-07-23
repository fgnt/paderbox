import numpy as np
import scipy

from scipy import signal

import pylab as plt
import seaborn as sns
sns.set_palette("deep", desat=.6)
cmap = sns.diverging_palette(220, 20, n=7, as_cmap=True)

from numpy.fft import rfft, irfft
#from scikits.talkbox import segment_axis

#def stft_jahn(x, frame_size, frame_shift, fft_size):
#    w = scipy.hamming(frame_size)
#    x = np.concatenate([x, np.zeros(frame_size-frame_shift)])
#    framed = segment_axis(x, frame_size, frame_size - frame_shift, end='cut') * w
#    return rfft(framed, fft_size, axis=-1)

#def istft_jahn(X, frame_size, frame_shift):
#    x = scipy.zeros(X.shape[0] * frame_shift)
#    for n, i in enumerate(range(0, len(x) - frame_size, frame_shift)):
#        x[i:i + frame_size] += np.real(irfft(X[n]))
#    return x

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


def stft_basj(x, fftsize=1024, overlap=4):
    hop = fftsize // overlap
    w = scipy.hanning(fftsize + 1)[:-1]      # better reconstruction with this trick +1)[:-1]
    return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])

def istft_basj(X, overlap=4):
    fftsize=(X.shape[1]-1)*2
    hop = fftsize // overlap
    w = scipy.hanning(fftsize+1)[:-1]
    x = scipy.zeros(X.shape[0]*hop)
    wsum = scipy.zeros(X.shape[0]*hop)
    for n,i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
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
