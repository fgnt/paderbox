import seaborn as sns
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from nt.speech_enhancement.beamform_utils import *
import nt.transform
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


def allow_dict_input_and_colorize(f):
    """ Allow dict input and use keys as labels
    """
    @wraps(f)
    def wrapper(signal, *args, **kwargs):
        ax = kwargs.pop('ax', None)

        if isinstance(signal, dict):
            # Scatter does not cycle the colors so we need to do this explicitly
            if f.__name__ == 'scatter':
                cyl = plt.rcParams['axes.prop_cycle']
                for (label, data), prob_cycle in zip(signal.items(), cyl):
                    ax = f(data, *args, ax=ax, label=label,
                           color=prob_cycle['color'], **kwargs)
            else:
                for label, data in signal.items():
                    ax = f(data, *args, ax=ax, label=label, **kwargs)
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
def line(*signal, ax=None, ylim=None, label=None, color=None, logx=False,
         logy=False):
    """
    Use together with facet_grid().

    Signal can be a dict with labels and data. Data can then be a tuple or
    a single vector of y-values.

    Example:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> plot.line(x, y)
        >> plot.line(y)
        >> plot.line((x, y))

    Example with faced grid:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> y2 = numpy.random.rand(100)
        >> facet_grid([y, y2], plot.line)
        >> facet_grid([y, (x, y2)], plot.line)
        # second figure with y over x
        >> facet_grid([y, [2, 1]], plot.line)
        # second figure with y values 2 and 1

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

    if len(signal) == 1 and isinstance(signal[0], tuple):
        signal = signal[0]

    if color is not None:
        plt_fcn(*signal, label=label, color=color)
    else:
        plt_fcn(*signal, label=label)

    if label is not None:
        ax.legend()

    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


@allow_dict_input_and_colorize
@create_subplot
def scatter(*signal, ax=None, ylim=None, label=None, color=None):
    """
    Use together with facet_grid().

    Signal can be a dict with labels and data. Data can then be a tuple or
    a single vector of y-values.

    Example:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> plot.scatter(x, y)
        >> plot.scatter(y)
        >> plot.scatter((x, y))

    Example with faced grid:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> y2 = numpy.random.rand(100)
        >> facet_grid([y, y2], plot.scatter)
        >> facet_grid([y, (x, y2)], plot.scatter)
        # second figure with y over x
        >> facet_grid([y, [2, 1]], plot.scatter)
        # second figure with y values 2 and 1

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """

    if len(signal) == 1 and isinstance(signal[0], tuple):
        signal = signal[0]

    if len(signal) == 1:
        signal = (range(len(signal[0])), signal[0])

    if color is not None:
        ax.scatter(*signal, label=label, color=color)
    else:
        ax.scatter(*signal, label=label)

    # if type(signal) is tuple:
    #     ax.scatter(signal[0], signal[1], label=label, color=color)
    # else:
    #     ax.scatter(range(len(signal)), signal, label=label, color=color)
    #
    # if ylim is not None:
    #     ax.set_ylim(ylim)

    return ax


@allow_dict_input_and_colorize
@create_subplot
def time_series(*signal, ax=None, ylim=None, label=None, color=None):
    """
    Use together with facet_grid().

    Signal can be a dict with labels and data. Data can then be a tuple or
    a single vector of y-values.

    Example:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> plot.time_series(x, y)
        >> plot.time_series(y)
        >> plot.time_series((x, y))

    Example with faced grid:

        >> x = numpy.random.rand(100)
        >> y = numpy.random.rand(100)
        >> y2 = numpy.random.rand(100)
        >> facet_grid([y, y2], plot.time_series)
        >> facet_grid([y, (x, y2)], plot.time_series)
        # second figure with y over x
        >> facet_grid([y, [2, 1]], plot.time_series)
        # second figure with y values 2 and 1

    :param signal: Single one-dimensional array or tuple of x and y values.
    :param ax: Axis handle
    :param ylim: Tuple with y-axis limits
    :return:
    """
    ax = line(*signal, ax=ax, ylim=ylim, label=label, color=color)
    if len(signal) == 2:
        ax.set_xlabel('Time / s')
    else:
        ax.set_xlabel('Sample index')

    ax.set_ylabel('Amplitude')
    return ax


def _time_frequency_plot(
        signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
        sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label=None, z_scale=None
):
    """

    :param signal:
    :param ax:
    :param limits: tuple: min, max, linthresh(only for symlog)
    :param log: transform to log
    :param colorbar:
    :param batch:
    :param sample_rate:
    :param stft_size:
    :param stft_shift:
    :param x_label:
    :param y_label:
    :param z_label:
    :param z_scale: how to scale the values ('linear', 'log', 'symlog' or instance of matplotlib.colors.Normalize)
    :return:
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

    if z_scale == 'linear' or z_scale == None:
        norm = None
    elif z_scale == 'log':
        norm = matplotlib.colors.LogNorm(
                    vmin=limits[0],
                    vmax=limits[1],)
    elif z_scale == 'symlog':
        if len(limits) == 2:
            # assume median is a good log2lin border
            # have anyone a better idea?
            limits = (*limits, np.median(np.abs(signal)))
        norm = matplotlib.colors.SymLogNorm(
                    linthresh=limits[2],
                    linscale=1,
                    vmin=limits[0],
                    vmax=limits[1],
                    clip=False,
                    )
    elif isinstance(z_scale, matplotlib.colors.Normalize):
        norm = z_scale
        if isinstance(z_scale, matplotlib.colors.SymLogNorm):
            z_scale = 'symlog'
    else:
        raise ValueError('z_scale: {} is invalid. '
                         'Expected: linear, log, symlog or instance of matplotlib.colors.Normalize'
                         ''.format(z_scale))

    image = ax.imshow(
        signal,
        interpolation='nearest',
        vmin=limits[0],
        vmax=limits[1],
        cmap='viridis',
        origin='lower',
        norm=norm,
    )

    if x_label is None:
        if sample_rate is not None and stft_shift is not None:
            seconds_per_tick = 0.5
            blocks_per_second = sample_rate/stft_shift
            x_tick_range = np.arange(0, signal.shape[1],
                                     seconds_per_tick*blocks_per_second)
            x_tick_labels = x_tick_range/blocks_per_second
            plt.xticks(x_tick_range, x_tick_labels)
            ax.set_xlabel('Time / s')
        else:
            ax.set_xlabel('Time frame index')
    else:
        ax.set_xlabel(x_label)

    if y_label is None:
        if sample_rate is not None and stft_size is not None:
            y_tick_range = np.linspace(0, stft_size/2, num=5)
            y_tick_labels = y_tick_range*(sample_rate/stft_size/1000)
            plt.yticks(y_tick_range, y_tick_labels)
            ax.set_ylabel('Frequency / kHz')
        else:
            ax.set_ylabel('Frequency bin index')
    else:
        ax.set_ylabel(y_label)

    if colorbar:
        if z_scale == 'symlog':
            # The default colorbar is not compatible with symlog scale.
            # Set the ticks to
            # max, ..., log2lin border, 0, log2lin border, ..., min
            tick_locations = (norm.vmin,
                              norm.vmin * 1e-1,
                              norm.vmin * 1e-2,
                              norm.vmin * 1e-3,
                              -norm.linthresh,
                              0.0,
                              norm.linthresh,
                              norm.vmax  * 1e-3,
                              norm.vmax  * 1e-2,
                              norm.vmax  * 1e-1,
                              norm.vmax)
            cbar = plt.colorbar(image, ax=ax, ticks=tick_locations)
        else:
            cbar = plt.colorbar(image, ax=ax)
        if z_label is None:
            if log:
                cbar.set_label('Energy / dB')
            else:
                cbar.set_label('Energy (linear)')
        else:
            cbar.set_label(z_label)

    ax.set_aspect('auto')
    ax.grid(False)
    return ax


@create_subplot
def spectrogram(
        signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
        sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label=None, z_scale=None
):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param signal: Real valued power spectrum
        with shape (frames, frequencies).
    :param limits: Color limits for clipping purposes.
    :param ax: Provide axis. I.e. for use with facet_grid().
    :param log: Take the logarithm of the signal before plotting
    :param colorbar: Display a colorbar right to the plot
    :param batch: If the decode has 3 dimensions:
        Specify the which batch to plot
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
    return _time_frequency_plot(**locals())


@create_subplot
def stft(
        signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
         sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label=None, z_scale=None,
):
    """
    Plots a spectrogram from an stft signal as input. This is a wrapper of the
    plot function for spectrograms.

    :param signal: Complex valued stft signal.
    :param limits: Color limits for clipping purposes.
    :param log: Take the logarithm of the signal before plotting
    :param colorbar: Display a colorbar right to the plot
    :param batch: If the decode has 3 dimensions:
        Specify the which batch to plot
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
    signal = nt.transform.stft_to_spectrogram(signal)
    return spectrogram(**locals())


@create_subplot
def mask(
        signal, ax=None, limits=(0, 1), log=False, colorbar=True, batch=0,
        sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label='Mask', z_scale=None
):
    """
    Plots any mask with values between zero and one.

    :param signal: Mask with shape (time-frames, frequency-bins)
    :param ax: Optional figure axis for use with facet_grid()
    :param limits: Clip the signal to these limits
    :param colorbar: Show colorbar right to the plot
    :param batch: If the decode has 3 dimensions:
        Specify the which batch to plot
    :return: axes
    """
    return _time_frequency_plot(**locals())


@create_subplot
def tf_symlog(
        signal, ax=None, limits=None, log=False, colorbar=True, batch=0,
        sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label='tf_symlog', z_scale='symlog'
):
    """
    Plots any time frequency data. limits will be symetric and z_scale symlog,
    where symlog means log scale outside the median

    :param signal: Mask with shape (time-frames, frequency-bins)
    :param ax: Optional figure axis for use with facet_grid()
    :param limits: Clip the signal to these limits
    :param colorbar: Show colorbar right to the plot
    :param batch: If the decode has 3 dimensions:
        Specify the which batch to plot
    :return: axes
    """
    limits = np.max(np.abs(signal))
    limits = (-limits, limits, np.median(np.abs(signal)))
    return _time_frequency_plot(**locals())


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
    # x = softmax(x)
    if label_handler is not None:
        ordered_map = OrderedDict(
            sorted(label_handler.int_to_label.items(), key=lambda t: t[1])
        )
        order = list(ordered_map.keys())
    else:
        order = list(range(x.shape[1]))

    image = ax.imshow(
        x[:, order].T,
        cmap='viridis', interpolation='none', aspect='auto'
    )
    cbar = plt.colorbar(image, ax=ax)

    if label_handler is not None:
        plt.yticks(
            range(len(label_handler)),
            list(ordered_map.values()))

    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Transcription')
    return ax
