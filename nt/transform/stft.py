import numpy as np
import scipy

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
