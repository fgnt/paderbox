import numpy as np
from paderbox.transform import stft, bark_fbank

def ams(time_signal, version = 1):
    """
    Compute AMS (Amplitude Modulation Spectrogram) Features from an audio signal.

    Features are 3-dimensional (time, center-frequency, modulation-frequency).

    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5947602

    :param time_signal:
    :return:
    """
    # STFT
    stft_signal = stft(time_signal)

    # Squaring magnitude of complex values
    stft_signal = stft_signal**2

    # Critical Bandwidth Analysis
    bark_signal = bark_fbank(time_signal)

    # Second STFT
    nframes, nbands = bark_signal.shape
    y = np.zeros((5, nbands, 513))
    for i in range(nbands):
        y[:, i, :] = stft(bark_signal[:, i])

    if version == 1:
        # abs & log
        y = np.abs(y)
        y = np.log(y)
    elif version == 2:
        # Cube root, real or im part, normalize to unit circle, multiply compressed length of complex pointer
        y = y**0.33 * np.real(y)/np.abs(y)

    return y
