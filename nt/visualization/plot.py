import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nt.speech_enhancement.beamform_utils import *
import nt.transform
from nt.utils.math_ops import softmax
from warnings import warn
from collections import OrderedDict
from functools import wraps


def create_subplot(f):
    """ This decorator creates a subplot and passes the axes object if needed.

    This function helps you to create figures in subplot notation, even when
    you are not using subplots. Use it as a decorator.

    :param f: Function to be wrapped
    :return: Axes object
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        ax = kwargs.pop('ax', None)
        if ax is None:
            figure, ax = plt.subplots(1, 1)
        title = kwargs.pop('title', None)
        if title is not None:
            ax.set_title(title)
        return f(*args, ax=ax, **kwargs)
    return wrapper


def _get_batch(signal, batch):
    if signal.ndim == 3:
        return signal[:, batch, :]
    elif signal.ndim == 2:
        return signal
    else:
        raise ValueError('The signal can only be two or three dimensional')


@create_subplot
def line(signal, ax=None, ylim=None):
    """
    Use together with facet_grid().

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    if type(signal) is tuple:
        ax.plot(signal[0], signal[1])
    else:
        ax.plot(signal)

    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


@create_subplot
def scatter(signal, ax=None, ylim=None):
    """
    Use together with facet_grid().

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    if type(signal) is tuple:
        ax.scatter(signal[0], signal[1])
    else:
        ax.scatter(range(len(signal)), signal)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


@create_subplot
def time_series(signal, ax=None, ylim=None):
    """
    Use together with facet_grid().

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    ax = line(signal, ax=ax, ylim=ylim)
    if type(signal) is tuple:
        ax.set_xlabel('Time / s')
    else:
        ax.set_xlabel('Sample index')

    ax.set_ylabel('Amplitude')
    return ax


@create_subplot
def spectrogram(signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
                sample_rate=None, stft_size=None, x_label='Time frame index',
                y_label='Frequency bin index'):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param signal: Real valued power spectrum
        with shape (frames, frequencies).
    :param limits: Color limits for clipping purposes.
    :param ax: Provide axis. I.e. for use with facet_grid().
    :param log: Take the logarithm of the signal before plotting
    :param colorbar: Display a colorbar right to the plot
    :param batch: If the decode has 3 dimensions: Specify the which batch to plot
    :param sample_rate: Sample rate of the signal.
        If `sample_rate` and `stft_size` are specified, the y-ticks will be the
        frequency
    :param stft_size: Size of the STFT transformation.
        If `sample_rate` and `stft_size` are specified, the y-ticks will be the
        frequency
    :return: axes
    """
    signal = _get_batch(signal, batch)

    if np.any(signal < 0) and log:
        warn('The array passed to spectrogram contained negative values. This '
             'leads to a wrong visualization and especially colorbar!')

    if log:
        signal = np.log10(signal).T
    else:
        signal = signal.T

    if limits is None:
        limits = (np.min(signal), np.max(signal))

    image = ax.imshow(signal,
                      interpolation='nearest',
                      vmin=limits[0], vmax=limits[1],
                      cmap='viridis', origin='lower')
    if colorbar:
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Energy / dB')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if sample_rate is not None and stft_size is not None:
        y_tick_range = np.linspace(0, 1024/2, num=5)
        y_tick_labels = y_tick_range*(sample_rate/stft_size/1000)
        plt.yticks(y_tick_range, y_tick_labels)
        ax.set_ylabel('Frequency / kHz')
    ax.grid(False)
    return ax


@create_subplot
def stft(signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
         sample_rate=None, stft_size=None):
    """
    Plots a spectrogram from an stft signal as input. This is a wrapper of the
    plot function for spectrograms.

    :param signal: Complex valued stft signal.
    :param limits: Color limits for clipping purposes.
    :param log: Take the logarithm of the signal before plotting
    :param colorbar: Display a colorbar right to the plot
    :param batch: If the decode has 3 dimensions: Specify the which batch to plot
    :param sample_rate: Sample rate of the signal.
        If `sample_rate` and `stft_size` are specified, the y-ticks will be the
        frequency
    :param stft_size: Size of the STFT transformation.
        If `sample_rate` and `stft_size` are specified, the y-ticks will be the
        frequency
    :return: axes
    """
    return spectrogram(nt.transform.stft_to_spectrogram(signal), limits=limits,
                       ax=ax, log=log, colorbar=colorbar, batch=batch,
                       sample_rate=sample_rate, stft_size=stft_size)


@create_subplot
def mask(signal, ax=None, limits=(0, 1), colorbar=True, batch=0):
    """
    Plots any mask with values between zero and one.

    :param signal: Mask with shape (time-frames, frequency-bins)
    :param ax: Optional figure axis for use with facet_grid()
    :param limits: Clip the signal to these limits
    :param colorbar: Show colorbar right to the plot
    :param batch: If the decode has 3 dimensions: Specify the which batch to plot
    :return: axes
    """

    signal = _get_batch(signal, batch)
    image = ax.imshow(signal.T,
                      interpolation='nearest', origin='lower',
                      vmin=limits[0], vmax=limits[1],
                      cmap='viridis')
    if colorbar:
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label('Mask')
    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Frequency bin index')
    ax.grid(False)
    return ax


@create_subplot
def plot_ctc_decode(decode, label_handler, ax=None, batch=0):
    """ Plot a ctc decode

    :param decode: Output of the network
    :param label_handler: The label handler
    :param ax: Optional figure axes to use with facet_grid()
    :param batch: If the decode has 3 dimensions: Specify the which batch to plot
    :return:
    """
    net_out = _get_batch(decode, batch)
    net_out -= np.amax(net_out)
    net_out_e = np.exp(net_out)
    net_out = net_out_e / (np.sum(net_out_e, axis=1, keepdims=True) + 1e-20)

    for char in range(decode.shape[2]):
        _ = ax.plot(net_out[:, char],
                    label=label_handler.int_to_label[char])
        plt.legend(loc='lower center',
                   ncol=decode.shape[2] // 3,
                   bbox_to_anchor=[0.5, -0.35])
    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Probability')
    return ax


@create_subplot
def plot_nn_current_loss(status, ax=None):
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
    return ax


@create_subplot
def plot_nn_current_loss_distribution(status, ax=None):
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
    return ax


@create_subplot
def plot_nn_current_timings_distribution(status, ax=None):
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
    return ax


@create_subplot
def plot_beampattern(W, sensor_positions, fft_size, sample_rate,
                     source_angles=None, ax=None):
    if source_angles is None:
        source_angles = numpy.arange(-numpy.pi, numpy.pi, 2 * numpy.pi / 360)
        source_angles = numpy.vstack(
            [source_angles, numpy.zeros_like(source_angles)]
        )

    tdoa = get_farfield_time_difference_of_arrival(
        source_angles,
        sensor_positions
    )
    s_vector = get_steering_vector(tdoa, fft_size, sample_rate)

    B = numpy.zeros((fft_size // 2, source_angles.shape[1]))
    for f in range(fft_size // 2):
        for k in range(source_angles.shape[1]):
            B[f, k] = numpy.abs(W[f].dot(s_vector[f, :, k])) ** 2 / \
                      numpy.abs(W[f].dot(W[f])) ** 2

    image = ax.imshow(10 * numpy.log10(B),
                      vmin=-10, vmax=10,
                      interpolation='nearest',
                      cmap='viridis', origin='lower')
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Gain / dB')
    ax.set_xlabel('Angle')
    ax.set_ylabel('Frequency bin index')
    ax.grid(False)
    return ax


@create_subplot
def plot_ctc_label_probabilities(net_out, ax=None, label_handler=None, batch=0):
    """ Plots a posteriorgram of the network output of a CTC trained network

    :param net_out: Output of the network
    :param label_handler: Labelhandler holding the correspondence labels
    :param batch: Batch to plot
    """
    x = _get_batch(net_out, batch)
    x = softmax(x)
    if label_handler is not None:
        ordered_map = OrderedDict(
            sorted(label_handler.int_to_label.items(), key=lambda t: t[1])
        )
        order = list(ordered_map.keys())
    else:
        order = list(range(x.shape[1]))

    ax.imshow(
        x[:, order].T,
        cmap='viridis', interpolation='none', aspect='auto'
    )

    if label_handler is not None:
        plt.yticks(
            range(len(label_handler)),
            list(ordered_map.values()))

    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Transcription')
    return ax
