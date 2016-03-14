import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nt.speech_enhancement.beamform_utils import *
import nt.transform
from nt.utils.math_ops import softmax
from warnings import warn
from collections import OrderedDict
from functools import wraps
from nt.visualization.new_cm import viridis_hex


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


def allow_dict_input_and_colorize(f):
    """ Allow dict input and use keys as labels
    """
    @wraps(f)
    def wrapper(signal, *args, **kwargs):
        ax = kwargs.pop('ax', None)

        if isinstance(signal, dict):
            colors = viridis_hex[::len(viridis_hex) // len(signal)]
            for (label, data), color in zip(signal.items(), colors):
                ax = f(data, *args, ax=ax, label=label, color=color, **kwargs)
            ax.legend()
        else:
            ax = f(signal, *args, ax=ax, **kwargs)
        return ax
    return wrapper


def _get_batch(signal, batch):
    if signal.ndim == 3:
        return signal[:, batch, :]
    elif signal.ndim == 2:
        return signal
    else:
        raise ValueError('The signal can only be two or three dimensional')


@allow_dict_input_and_colorize
@create_subplot
def line(signal, ax=None, ylim=None, label=None, color=None, logx=False,
         logy=False):
    """
    Use together with facet_grid().

    Signal can be a dict with labels and data. Data can then be a tuple or
    a single vector of y-values.

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    if logx and logy:
        plt_fcn = ax.loglog
    elif logx:
        plt_fcn = ax.semilogx
    elif logy:
        plt_fcn = ax.semilogy
    else:
        plt_fcn = ax.plot

    if isinstance(signal, tuple):
        plt_fcn(signal[0], signal[1], label=label, color=color)
    else:
        plt_fcn(signal, label=label, color=color)

    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


@allow_dict_input_and_colorize
@create_subplot
def scatter(signal, ax=None, ylim=None, label=None, color=None):
    """
    Use together with facet_grid().

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    if type(signal) is tuple:
        ax.scatter(signal[0], signal[1], label=label, color=color)
    else:
        ax.scatter(range(len(signal)), signal, label=label, color=color)

    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


@allow_dict_input_and_colorize
@create_subplot
def time_series(signal, ax=None, ylim=None, label=None, color=None):
    """
    Use together with facet_grid().

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    ax = line(signal, ax=ax, ylim=ylim, label=label, color=None)
    if type(signal) is tuple:
        ax.set_xlabel('Time / s')
    else:
        ax.set_xlabel('Sample index')

    ax.set_ylabel('Amplitude')
    return ax


@create_subplot
def spectrogram(signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
                sample_rate=None, stft_size=None, stft_shift=None,
                x_label='Time frame index',
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
    :param stft_shift: Amount of samples the stft window is advanced per frame.
        If `sample_rate` and `stft_shift` are specified, the x-ticks will be the
        time
    :return: axes
    """
    signal = _get_batch(signal, batch)

    if np.any(signal < 0) and log:
        warn('The array passed to spectrogram contained negative values. This '
             'leads to a wrong visualization and especially colorbar!')

    if log:
        signal = np.log10(np.maximum(signal, np.max(signal)/1e6)).T
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
        if log:
            cbar.set_label('Energy / dB')
        else:
            cbar.set_label('Energy (linear)')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if sample_rate is not None and stft_size is not None:
        y_tick_range = np.linspace(0, stft_size/2, num=5)
        y_tick_labels = y_tick_range*(sample_rate/stft_size/1000)
        plt.yticks(y_tick_range, y_tick_labels)
        ax.set_ylabel('Frequency / kHz')
    if sample_rate is not None and stft_shift is not None:
        seconds_per_tick = 0.5
        blocks_per_second = sample_rate/stft_shift
        x_tick_range = np.arange(0, signal.shape[1],
                                 seconds_per_tick*blocks_per_second)
        x_tick_labels = x_tick_range/blocks_per_second
        plt.xticks(x_tick_range, x_tick_labels)
        ax.set_xlabel('Time / s')
    ax.grid(False)
    return ax


@create_subplot
def stft(signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
         sample_rate=None, stft_size=None, stft_shift=None):
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
    :param stft_shift: Amount of samples the stft window is advanced per frame.
        If `sample_rate` and `stft_shift` are specified, the x-ticks will be the
        time
    :return: axes
    """
    return spectrogram(nt.transform.stft_to_spectrogram(signal), limits=limits,
                       ax=ax, log=log, colorbar=colorbar, batch=batch,
                       sample_rate=sample_rate, stft_size=stft_size,
                       stft_shift=stft_shift)


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
