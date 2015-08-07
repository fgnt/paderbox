import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import nt.transform

COLORMAP = sns.diverging_palette(220, 20, n=7, as_cmap=True)

def time_series(signal, ax):
    """
    Use together with facet_grid().

    :param f: Figure handle
    :param ax: Axis handle
    :param x: Tuple with time indices as first, and data as second element.
    :return:
    """

    with sns.axes_style("darkgrid"):
        if type(signal) is tuple:
            ax.plot(signal[0], signal[1])
            ax.set_xlabel('Time / s')
        else:
            ax.plot(signal)
            ax.set_xlabel('Sample index')
        ax.set_ylabel('Amplitude')
        ax.grid(True)


def spectrogram(signal, limits=None, ax=None):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param signal: Real valued power spectrum
        with shape (frames, frequencies).
    :param limits: Color limits for clipping purposes.
    :param ax: Provide axis. I.e. for use with facet_grid().
    :return: None
    """
    signal = np.log10(signal).T

    if limits is None:
        limits = (np.min(signal), np.max(signal))

    with sns.axes_style("dark"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        image = ax.imshow(np.clip(signal, limits[0], limits[1]),
                          interpolation='nearest',
                          cmap=COLORMAP, origin='lower')
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Energy / dB')
        ax.set_xlabel('Time frame index')
        ax.set_ylabel('Frequency bin index')
        ax.grid(False)

def stft(signal, limits=None, ax=None):
    """
    Plots a spectrogram from an stft signal as input. This is a wrapper of the
    plot function for spectrograms.

    :param signal: Complex valued stft signal.
    :param limits: Color limits for clipping purposes.
    :return: None
    """
    spectrogram(nt.transform.stft_to_spectrogram(signal), limits=limits, ax=ax)

def mask(signal, ax=None, **kwargs):
    """
    Plots any mask with values between zero and one.

    :param signal: Mask with shape (time-frames, frequency-bins)
    :param ax: Optional figure axis for use with facet_grid()
    :return:
    """
    limits = (0, 1)

    with sns.axes_style("dark"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        image = ax.imshow(np.clip(signal.T, limits[0], limits[1]),
                          interpolation='nearest', origin='lower')
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Mask')
        ax.set_xlabel('Time frame index')
        ax.set_ylabel('Frequency bin index')
        ax.grid(False)

def plot_ctc_decode(decode, label_handler, ax=None):
    """ Plot a ctc decode

    :param decode: Output of the network
    :param label_handler: The label handler
    :param ax: Optional figure axes to use with facet_grid()
    :return:
    """

    with sns.axes_style("darkgrid"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
    for char in range(decode.shape[2]):
        _ = ax.plot(decode[:, 0, char], label=label_handler.int_to_label[char])
        plt.legend(loc='lower center',
                   ncol=decode.shape[2]//3,
                   bbox_to_anchor=[0.5, -0.35])
    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Unnormalized Propability')
