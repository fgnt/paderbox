from collections import OrderedDict
from functools import wraps
import itertools
from warnings import warn

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import paderbox.transform


__all__ = [
    'seq2seq_alignment',
    'beampattern',
    'tf_symlog',
    'phone_alignment',
    'mask',
    'stft',
    'spectrogram',
    'time_series',
    'scatter',
    'line',
    'image'
]


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
            _, ax = plt.subplots(1, 1)
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
            for label, data in signal.items():
                ax = f(data, *args, ax=ax, label=label, **kwargs)
            ax.legend()
        else:
            ax = f(signal, *args, ax=ax, **kwargs)
        return ax
    return wrapper


def allow_dict_for_title(f):
    """ Allow dict input and use keys as labels
    """
    @wraps(f)
    def wrapper(signal, *args, **kwargs):
        ax = kwargs.pop('ax', None)

        if isinstance(signal, dict):
            for label, data in signal.items():
                ax = f(data, *args, ax=ax, title=label, **kwargs)
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
        raise ValueError(f'The signal can only be two or three dimensional. '
                         f'Shape: {signal.shape}')


def _xy_plot(
        *signal, plt_fcn,
        ax=None,
        xlim=None, ylim=None,
        label=None,
        color=None,
        xlabel=None, ylabel=None,
        **kwargs
):
    # ToDo: Use this function for line, scatter and stem
    if len(signal) == 1 and isinstance(signal[0], tuple):
        signal = signal[0]

    if color is not None:
        plt_fcn(*signal, label=label, color=color, **kwargs)
    else:
        plt_fcn(*signal, label=label, **kwargs)

    if label is not None:
        ax.legend()

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax


@allow_dict_input_and_colorize
@create_subplot
def stem( # pylint: disable=unused-argument
        *signal,
        ax=None,
        xlim=None, ylim=None,
        label=None,
        color=None,
        xlabel=None, ylabel=None,
        **kwargs
):
    l = locals()
    l.pop('kwargs')
    l.pop('signal')
    return _xy_plot(*signal, plt_fcn=ax.stem, **l, **kwargs)


@allow_dict_input_and_colorize
@create_subplot
def line(*signal, ax: plt.Axes = None, xlim=None, ylim=None, label=None,
         color=None, logx=False, logy=False, xlabel=None, ylabel=None,
         **kwargs):
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
        plt_fcn(*signal, label=label, color=color, **kwargs)
    else:
        plt_fcn(*signal, label=label, **kwargs)

    if label is not None:
        #pylint: disable=protected-access
        if ax.get_legend() and ax.get_legend()._legend_title_box.get_visible():
            title = ax.get_legend().get_title().get_text()
        else:
            title = None
        ax.legend(title=title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return ax


@allow_dict_input_and_colorize
@create_subplot
def scatter(*signal, ax=None, ylim=None, label=None, color=None, zorder=None,
            marker=None, xlim=None, xlabel=None, ylabel=None,
            logx=False, logy=False,
            **kwargs):
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
    kwargs['marker'] = marker

    if len(signal) == 1 and isinstance(signal[0], tuple):
        signal = signal[0]

    if len(signal) == 1:
        signal = (range(len(signal[0])), signal[0])

    if zorder is not None:
        kwargs['zorder'] = zorder
    if color is None:
        # Scatter does not cycle the colors so we need to do this explicitly
        # pylint: disable=protected-access
        color = ax._get_lines.get_next_color()

    if color is not None:
        if isinstance(color, (list, tuple, np.ndarray, itertools.count)):
            if isinstance(color, itertools.count):
                color = list(itertools.islice(color, len(signal[0])))

            ax.scatter(*signal, label=label, c=color, **kwargs)
        else:
            ax.scatter(*signal, label=label, color=color, **kwargs)
    else:
        ax.scatter(*signal, label=label, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # if type(signal) is tuple:
    #     ax.scatter(signal[0], signal[1], label=label, color=color)
    # else:
    #     ax.scatter(range(len(signal)), signal, label=label, color=color)

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_xscale('log')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if label is not None:
        ax.legend()

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
        x_label=None, y_label=None, z_label=None, z_scale=None, cmap=None,
        cbar_ticks=None, cbar_tick_labels=None, xticks=None, xtickslabels=None,
        origin='lower'
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
    :param z_scale: how to scale the values
        ('linear', 'log', 'symlog' or instance of matplotlib.colors.Normalize)
    :return:
    """

    signal = _get_batch(signal, batch)

    if log and np.any(signal < 0):
        warn('The array passed to spectrogram contained negative values. This '
             'leads to a wrong visualization and especially colorbar!')

    if log:
        signal = 10 * np.log10(np.maximum(signal, np.max(signal) / 1e6)).T
    else:
        signal = signal.T

    if limits is None:
        limits = (np.min(signal), np.max(signal))

    if cmap is None:
        cmap = 'viridis'

    if z_scale == 'linear' or z_scale is None:
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
        raise ValueError(
            f'z_scale: {z_scale} is invalid.\n'
            'Expected: linear, log, symlog or instance of '
            'matplotlib.colors.Normalize'
        )

    image_ = ax.imshow(
        signal,
        interpolation='nearest',
        vmin=limits[0],
        vmax=limits[1],
        cmap=cmap,
        origin=origin,
        norm=norm,
    )

    if x_label is None:
        if sample_rate is not None and stft_shift is not None:
            seconds_per_tick = 0.5
            blocks_per_second = sample_rate / stft_shift
            x_tick_range = np.arange(0, signal.shape[1],
                                     seconds_per_tick * blocks_per_second)
            x_tick_labels = x_tick_range / blocks_per_second
            plt.xticks(x_tick_range, x_tick_labels)
            ax.set_xlabel('Time / s')
        else:
            ax.set_xlabel('Time frame index')
    else:
        ax.set_xlabel(x_label)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xtickslabels is not None:
        ax.set_xticklabels(xtickslabels)

    if y_label is None:
        if sample_rate is not None and stft_size is not None:
            y_tick_range = np.linspace(0, stft_size / 2, num=5)
            y_tick_labels = y_tick_range * (sample_rate / stft_size / 1000)
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
            if cbar_ticks is None:
                tick_locations = (norm.vmin,
                                  norm.vmin * 1e-1,
                                  norm.vmin * 1e-2,
                                  norm.vmin * 1e-3,
                                  -norm.linthresh,
                                  0.0,
                                  norm.linthresh,
                                  norm.vmax * 1e-3,
                                  norm.vmax * 1e-2,
                                  norm.vmax * 1e-1,
                                  norm.vmax)
            else:
                tick_locations = cbar_ticks
            cbar = plt.colorbar(image_, ax=ax, ticks=tick_locations)
        else:
            cbar = plt.colorbar(image_, ax=ax, ticks=cbar_ticks)

        if cbar_tick_labels is not None:
            cbar.ax.set_yticklabels(cbar_tick_labels)

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
def spectrogram( # pylint: disable=unused-argument
        signal, ax=None, limits=None, log=True, colorbar=True, batch=0,
        sample_rate=None, stft_size=None, stft_shift=None,
        x_label=None, y_label=None, z_label=None, z_scale=None, cmap=None,
        cbar_ticks=None, cbar_tick_labels=None
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
def image( # pylint: disable=unused-argument
        signal, ax=None, x_label='', y_label='', z_label='', cmap=None,
        colorbar=False
):
    """
    Plots a spectrogram from a spectrogram (power) as input.

    :param signal: Real valued power spectrum
        with shape (frames, frequencies).
    :param ax: Provide axis. I.e. for use with facet_grid().
    :return: axes
    """
    signal = signal.T
    return _time_frequency_plot(
        limits=None, log=False, origin='upper', **locals()
    )


@create_subplot
def stft( # pylint: disable=unused-argument
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
    signal = paderbox.transform.stft_to_spectrogram(signal)
    return spectrogram(**locals())


@allow_dict_for_title
@create_subplot
def mask( # pylint: disable=unused-argument
        signal, ax=None, limits=(0, 1), log=False,
        colorbar=True, batch=0, sample_rate=None, stft_size=None,
        stft_shift=None, x_label=None, y_label=None, z_label='Mask',
        z_scale=None, cmap=None,
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


def _switch_indices(alignment):
    return np.concatenate(
        [[0], np.argwhere(alignment[:-1] != alignment[1:])[:, 0]])


@allow_dict_for_title
@create_subplot
def phone_alignment( # pylint: disable=unused-argument
        signal, phone_alignment, ax=None, limits=None, log=True,
        colorbar=True, batch=0, sample_rate=None, stft_size=None,
        stft_shift=None, x_label=None, y_label=None, z_label=None, z_scale=None
):
    """
    Plots any mask with values between zero and one.

    Example (using tf_speech):

        >> from tf_speech.notebook import *
        >> db = dispatch_db('reverb')
        >> sample = fetcher.get_sample()
        >> utt_id = sample['features']['utt_id'][0].decode('utf-8')
        >> ali = np.array([db.id2phone(p) for p in db.phone_alignment[utt_id]])
        >> with context_manager(figure_size=(150, 10)):
        >>     plot.phone_alignment(sample['features'][keys.Y][0], ali)

    :param signal: Mask with shape (time-frames, frequency-bins)
    :param phone_alignment: Alignment vector (time-frames)
    :param ax: Optional figure axis for use with facet_grid()
    :param limits: Clip the signal to these limits
    :param colorbar: Show colorbar right to the plot
    :param batch: If the decode has 3 dimensions:
        Specify the which batch to plot
    :return: axes
    """
    xticks = (_switch_indices(phone_alignment)[1:] -
              _switch_indices(phone_alignment)[:-1]) // 2 +\
        _switch_indices(phone_alignment)[:-1]
    xtickslabels = phone_alignment[xticks]
    ax.vlines(_switch_indices(phone_alignment), 0, signal.shape[1],
              linestyle='dotted', color='r', alpha=.75)
    del phone_alignment
    signal = paderbox.transform.stft_to_spectrogram(signal)
    ax = _time_frequency_plot(**locals())
    return ax


def _get_limits_for_tf_symlog(signal, limits):
    if limits is None:
        signal_abs = np.abs(signal)
        maximum = signal_abs[np.isfinite(signal_abs)].max()
        if maximum == 0:
            maximum = 1e-12
        limits = (-maximum, maximum)
    if len(limits) == 2:
        linthresh = np.nanmedian(signal_abs)
        if linthresh == 0:
            nonzero = np.nonzero([0, 0])
            if len(nonzero[0]) > 0:
                linthresh = np.nanmin(signal_abs[nonzero])
            else:
                linthresh = 1e-14
        limits = (*limits, linthresh)
    return limits


@allow_dict_for_title
@create_subplot
def tf_symlog( # pylint: disable=unused-argument
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
    if limits is None:
        limits = _get_limits_for_tf_symlog(signal, limits)

    return _time_frequency_plot(**locals())


@create_subplot
def labeled_line_plot(probabilities, label_handler=None, ax=None, batch=0):
    """ Plots a line plot with optional labels from a label handler.

    :param probabilities: Probabilities
    :param label_handler: The label handler
    :param ax: Optional figure axes to use with facet_grid()
    :param batch: If the decode has 3 dimensions: Specify the which batch to plot
    :return:
    """
    probabilities = _get_batch(probabilities, batch)

    def _get_label(char):
        if label_handler is not None:
            return f'{label_handler.int_to_label[char]} [{char}]'
        else:
            return str(char)

    for char in range(probabilities.shape[-1]):
        _ = ax.plot(probabilities[:, char], label=_get_label(char))
    plt.legend(loc='lower center',
               ncol=probabilities.shape[-1] // 3,
               bbox_to_anchor=[0.5, -0.35])
    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Probability')
    return ax


@create_subplot
def beampattern(W, sensor_positions, fft_size, sample_rate,
                source_angles=None, ax=None, resolution=360):
    from pb_bss.extraction.beamform_utils import (
        get_steering_vector,
        get_farfield_time_difference_of_arrival,
    )

    if source_angles is None:
        source_angles = np.arange(-np.pi, np.pi, 2 * np.pi / resolution)
        source_angles = np.vstack(
            [source_angles, np.zeros_like(source_angles)]
        )

    tdoa = get_farfield_time_difference_of_arrival(
        source_angles,
        sensor_positions
    )
    s_vector = get_steering_vector(tdoa, fft_size, sample_rate)

    B = np.einsum('ab,bca->ac', W, s_vector) ** 2
    B /= np.einsum('ab,ab->a', W, W.conj())[:, None]
    B = np.abs(B)

    image_ = ax.imshow(10 * np.log10(B),
                       vmin=-10, vmax=10,
                       interpolation='nearest',
                       cmap='viridis', origin='lower')
    cbar = plt.colorbar(image_, ax=ax)
    cbar.set_label('Gain / dB')
    ax.set_xlabel('Angle')
    ax.set_ylabel('Frequency bin index')
    ax.grid(False)
    return ax


@create_subplot
def posteriorgram(
        probabilities, ax=None, label_handler=None, batch=0, blank_idx=None):
    """ Plots a posteriorgram

    :param probabilities: Probabilities for the labels
    :param label_handler: Labelhandler holding the correspondence labels
    :param batch: Batch to plot
    """
    x = _get_batch(probabilities, batch)
    if label_handler is not None:
        ordered_map = OrderedDict(
            sorted(label_handler.int_to_label.items(), key=lambda t: t[1])
        )
        order = list(ordered_map.keys())
        if blank_idx is not None:
            if blank_idx < 0:
                blank_idx += x.shape[-1]
            if len(order) < x.shape[-1]:
                order += [blank_idx]
                ordered_map[blank_idx] = '<BLANK>'
    else:
        order = list(range(x.shape[1]))

    image_ = ax.imshow(
        x[:, order].T,
        cmap='viridis', interpolation='none', aspect='auto'
    )
    _ = plt.colorbar(image_, ax=ax)

    if label_handler is not None:
        plt.yticks(
            range(len(ordered_map)),
            list(ordered_map.values()))

    ax.set_xlabel('Time frame index')
    ax.set_ylabel('Transcription')
    ax.grid(False, axis='x')
    ax.grid(linestyle='-.', axis='y')
    return ax


@create_subplot
def seq2seq_alignment(alignment, targets=None, decode=None, ax=None,
                      alignment_length=None, label_length=None,
                      label_handler=None, batch=None):
    """ Plots a posteriorgram of the network output of a CTC trained network

    :param net_out: Output of the network
    :param label_handler: Labelhandler holding the correspondence labels
    :param batch: Batch to plot
    """

    if batch is not None:
        alignment = alignment[:, batch, :]
        if alignment_length is not None:
            alignment = alignment[:alignment_length[batch]]
        if targets is not None:
            targets = targets[batch]
            if label_length is not None:
                targets = targets[:label_length[batch]]
        if decode is not None:
            decode = decode[batch]
            if label_length is not None:
                decode = decode[:label_length[batch]]

    ax.imshow(
        alignment,
        cmap='bone', interpolation='none', aspect='auto', origin='lower'
    )

    if label_handler is not None:
        if decode is not None:
            decode_labels = label_handler.ints2labels(
                np.argmax(decode, axis=1))
            if targets is not None:
                target_labels = label_handler.ints2labels(
                    targets)[:len(decode_labels)]
            else:
                target_labels = len(decode_labels) * ['']
        plt.yticks(
            range(len(decode_labels)),
            ['{} [{}]'.format(d, t)
             for d, t in zip(decode_labels, target_labels)]
        )
    ax.grid(False)
    ax.set_xlabel('frame index')
    ax.set_ylabel('decode')
    return ax
