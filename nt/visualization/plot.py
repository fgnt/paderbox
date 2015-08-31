import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nt.visualization.new_cm import cmaps
from nt.speech_enhancement.beamform_utils import *
import nt.transform

COLORMAP = cmaps['viridis']


def time_series(signal, ax, ylim=None):
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
        if ylim is not None:
            ax.set_ylim(ylim)


def spectrogram(signal, limits=None, ax=None, **kwargs):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param signal: Real valued power spectrum
        with shape (frames, frequencies).
    :param limits: Color limits for clipping purposes.
    :param ax: Provide axis. I.e. for use with facet_grid().
    :return: None
    """

    if kwargs.get('log', True):
        signal = np.log10(signal).T
    else:
        signal = signal.T

    if limits is None:
        limits = (np.min(signal), np.max(signal))

    with sns.axes_style("dark"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        image = ax.imshow(signal,
                          interpolation='nearest',
                          vmin=limits[0], vmax=limits[1],
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

    limits = kwargs.get('limits', (0, 1))

    with sns.axes_style("dark"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        image = ax.imshow(np.clip(signal.T, limits[0], limits[1]),
                          interpolation='nearest', origin='lower',
                          cmap=COLORMAP)
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
    if decode.ndim == 3:
        net_out = decode[:, 0, :]
    else:
        net_out = decode
    net_out = net_out - np.amax(net_out)
    net_out_e = np.exp(net_out)
    net_out = net_out_e / \
              (np.sum(net_out_e, axis=1, keepdims=True) + 1e-20)

    with sns.axes_style("darkgrid"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        for char in range(decode.shape[2]):
            _ = ax.plot(net_out[:, char],
                        label=label_handler.int_to_label[char])
            plt.legend(loc='lower center',
                       ncol=decode.shape[2] // 3,
                       bbox_to_anchor=[0.5, -0.35])
        ax.set_xlabel('Time frame index')
        ax.set_ylabel('Propability')


def plot_nn_current_loss(status, ax=None):
    with sns.axes_style("darkgrid"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        plot = False

        if len(status.loss_current_batch_training) > 1:
            ax.plot(status.loss_current_batch_training,
                    label='training')
            plot = True
        if len(status.loss_current_batch_cv) > 1:
            ax.plot(status.loss_current_batch_cv,
                    label='cross-validation')
            plot = True
        if plot:
            ax.set_xlabel('Iterations')
            ax.set_title('Batch loss')
            plt.legend()


def plot_nn_current_loss_distribution(status, ax=None):
    with sns.axes_style("darkgrid"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        plot = False

        if len(status.loss_current_batch_training) > 10:
            sns.distplot(status.loss_current_batch_training,
                         label='training', ax=ax)
            plot = True
        if len(status.loss_current_batch_cv) > 10:
            sns.distplot(status.loss_current_batch_cv,
                         label='cross-validation', ax=ax)
            plot = True
        if plot:
            ax.set_xlabel('Loss')
            ax.set_title('Probability')
            plt.legend()


def plot_nn_current_timings_distribution(status, ax=None):
    with sns.axes_style("darkgrid"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        plot = False

        if len(status.cur_time_forward) > 10:
            sns.distplot(status.cur_time_forward,
                         label='training forward', ax=ax)
            plot = True
        if len(status.cur_time_backprop) > 10:
            sns.distplot(status.cur_time_backprop,
                         label='training backprop', ax=ax)
            plot = True
        if len(status.cur_time_cv) > 10:
            sns.distplot(status.cur_time_cv,
                         label='cross-validation', ax=ax)
            plot = True
        if plot:
            ax.set_xlabel('Time [ms]')
            ax.set_title('Probability')
            plt.legend()


def plot_beampattern(W, sensor_positions, fft_size, sample_rate,
                     source_angles=None, ax=None):
    if source_angles is None:
        source_angles = numpy.arange(-numpy.pi, numpy.pi, 2 * numpy.pi / 360)
        source_angles = numpy.vstack([source_angles,
                                      numpy.zeros_like(source_angles)])

    tdoa = get_farfield_TDOA(source_angles, sensor_positions)
    s_vector = steering_vector(tdoa, fft_size, sample_rate)

    B = numpy.zeros((fft_size // 2, source_angles.shape[1]))
    for f in range(fft_size // 2):
        for k in range(source_angles.shape[1]):
            B[f, k] = numpy.abs(W[f].dot(s_vector[f, :, k])) ** 2 / \
                      numpy.abs(W[f].dot(W[f])) ** 2

    with sns.axes_style("dark"):
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        image = ax.imshow(10 * numpy.log10(B),
                          vmin=-10, vmax=10,
                          interpolation='nearest',
                          cmap=COLORMAP, origin='lower')
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Gain / dB')
        ax.set_xlabel('Angle')
        ax.set_ylabel('Frequency bin index')
        ax.grid(False)
