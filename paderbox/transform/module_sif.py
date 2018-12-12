import numpy as np
from paderbox.transform.module_stft import stft


def sif(time_signal, stft_size=512, stft_shift=160, window_length=None, num_features=52, denoise=True,
        append_energy=False):
    """
    Calculates the spectrogram image features from a given time signal. The implementation follows from the paper:
    Robust Audio Event Recognition with 1-Max Pooling Convolutional Neural Networks by
    Huy Phan, Lars Hertel, Marco Maass, Alfred Mertins.
    It appends frame energy if required.

    :param time_signal: Single channel time signal.
    :param stft_size: Scalar FFT-size.
    :param stft_shift: Scalar FFT-shift.
    :param window_length: Length of the hamming window, if different from FFT-size.
    :param num_features: Number of desired frequency bins. F in paper
    :param denoise:  Default is True.
    :param append_energy: Boolean. Adds energy of the frame if True.
    :return: Spectrogram image features with dimensions frames times num_features ( append_energy = False)
            OR frames times num_features+1 (append_energy = True)
    """

    a_stft = stft(time_signal, size=stft_size, shift=stft_shift, window_length=window_length,
                  fading=False)
    # Magnitude spectrogram
    a_stft = np.abs(a_stft)

    # Down sample frequency bins to num_features
    averaging_window_length = stft_size // (2 * num_features)
    ds_a_stft = list()
    for i in range(a_stft.shape[0]):
        row = [np.average(a_stft[i, j * averaging_window_length:(j + 1) * averaging_window_length]) for j in
               range(num_features)]
        ds_a_stft.append((np.asarray(row, dtype=np.float32)).reshape(1, len(row)))
    ds_a_stft = np.concatenate(ds_a_stft, axis=0)

    # Denoise the downsampled stft
    if denoise:
        ds_a_stft -= np.min(ds_a_stft, axis=0)

    if append_energy:
        energy = np.sum(ds_a_stft, axis=1).reshape((-1, 1))
        ds_a_stft = np.concatenate((ds_a_stft, energy), axis=1)

    return ds_a_stft
