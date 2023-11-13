import typing
import numpy as np
import paderbox as pb


__all__ = [
    'np_kernel',
    'ai_dilate',
    'ai_erode',
    'max_kernel',
    'min_kernel',
    'mean_kernel',
    'median_kernel',
]


def _plot(a):
    """
    >>> _plot([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ' ▁▂▃▄▅▆▇█'
    """
    a = np.array(a)
    values = np.array(list(' ▁▂▃▄▅▆▇█'))
    m = np.amax(a)
    if m == 0:
        m = 1
    a = np.round(a * ((len(values)-1) / m)).astype(int)
    if a.ndim == 1:
        print(repr(''.join(values[a])))
    elif a.ndim == 2:
        for e in a:
            print(repr(''.join(values[e])))
    else:
        raise Exception(a.shape)


def np_kernel1d(
        x,
        kernel_size,  # an odd number
        *,
        axis=-1,
        kernel: callable,  # e.g. np.mean, np.amax, np.amin
        padding_mode: 'typing.Literal["edge"]' = 'edge',
        pad_position: 'typing.Literal["pre", "post", None]' = 'pre',
):
    """
    Apply a kernel to the last axis of x.

    Args:
        x: np.array
        kernel_size:
        axis:
        kernel: a function that should be applied to the kernel window.
            Has to accept an axis argument, like np.mean, np.amax, np.median.
        padding_mode:
            The mode for np.pad, see np.pad for the options.
        pad_position:
            'pre': First pad and then apply the kernel.
            'post': First apply the kernel then pad.
            None: Don't pad and return an array with a smaller last axis.

    Returns:

    >>> a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,])
    >>> _plot(a); _plot(np_kernel1d(a, 3, kernel=np.amax))
    '    ████    '
    '   ██████   '
    >>> _plot(a); _plot(np_kernel1d(a, 3, kernel=np.amin))
    '    ████    '
    '     ██     '
    >>> _plot(a); _plot(np_kernel1d(a, 3, kernel=np.mean))
    '    ████    '
    '   ▃▅██▅▃   '

    >>> a = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,]).reshape(3, 4)
    >>> _plot(a)
    '    '
    ' ██ '
    '   █'
    >>> _plot(np_kernel1d(a, 3, kernel=np.mean))
    '    '
    '▄██▄'
    '  ▄█'
    >>> _plot(np_kernel1d(a, 3, kernel=np.mean, axis=1))
    '    '
    '▄██▄'
    '  ▄█'
    >>> _plot(np_kernel1d(a.T, 3, kernel=np.mean, axis=0).T)
    '    '
    '▄██▄'
    '  ▄█'
    >>> _plot(np_kernel1d(a.T, 3, kernel=np.mean, axis=-2).T)
    '    '
    '▄██▄'
    '  ▄█'

    """
    assert kernel_size % 2 == 1, (kernel_size, 'kernel size has to be odd.')
    assert pad_position in ['pre', 'post', None], pad_position

    if 0 <= axis < x.ndim:
        # segment_axis adds an axis, a negative axis allows to use the same
        # axis for segment_axis and kernel
        axis = axis - x.ndim

    if pad_position == 'pre':
        shift = kernel_size // 2
        x = pb.array.pad_axis(x, (shift, shift), axis=axis, mode=padding_mode)

    y = kernel(
        pb.array.segment_axis(x, kernel_size, 1, axis=axis, end='pad'),
        axis=axis)

    if pad_position == 'post':
        shift = kernel_size // 2
        y = pb.array.pad_axis(y, (shift, shift), axis=axis, mode=padding_mode)
    return y


def _ai_dilate_erode(ai, khalf):
    """


    # Test inverse mode an unknown shape
    >>> a = pb.array.interval.ones()
    >>> a[3:7] = False
    >>> _plot(a[:10]); _ = [_plot(ai_dilate(a, k)[:10]) for k in [1, 3, 5, 11]]
    '███    ███'
    '███    ███'
    '████  ████'
    '██████████'
    '██████████'

    """
    pairs = np.array(ai.normalized_intervals)
    assert pairs.shape[-1] == 2, pairs.shape
    if ai.inverse_mode:
        khalf = -khalf
    pairs[:, 0] -= khalf
    pairs[:, 1] += khalf
    pairs = np.maximum(pairs, 0)
    if ai.shape is not None:
        pairs = np.minimum(pairs, ai.shape[-1])
    return pb.array.interval.core.ArrayInterval.from_pairs(
        pairs.tolist(), shape=ai.shape, inverse_mode=ai.inverse_mode)


def ai_dilate(ai, kernel_size):
    """
    >>> import paderbox as pb
    >>> a = pb.array.interval.ArrayInterval(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,], dtype=bool))
    >>> a
    ArrayInterval("4:8", shape=(12,))
    >>> _plot(a); _ = [_plot(ai_dilate(a, k)) for k in [1, 3, 5, 11]]
    '    ████    '
    '    ████    '
    '   ██████   '
    '  ████████  '
    '████████████'


    """
    assert kernel_size % 2 == 1, (kernel_size, 'kernel size has to be odd.')
    return _ai_dilate_erode(ai, kernel_size//2)


def ai_erode(ai, kernel_size):
    """

    >>> import paderbox as pb
    >>> a = pb.array.interval.ArrayInterval(np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,], dtype=bool))
    >>> a
    ArrayInterval("4:8", shape=(12,))
    >>> _plot(a); _ = [_plot(ai_erode(a, k)) for k in [1, 3, 11]]
    '    ████    '
    '    ████    '
    '     ██     '
    '            '
    """
    assert kernel_size % 2 == 1, (kernel_size, 'kernel size has to be odd.')
    return _ai_dilate_erode(ai, -(kernel_size//2))


def max_kernel1d(x, kernel_size):
    """
    Apply a max kernel to x.
    In case of a boolean arrays, this operation is known as dilation.

    >>> a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,])
    >>> _plot(a); _plot(max_kernel1d(a, 3))
    '    ████    '
    '   ██████   '
    """
    if isinstance(x, np.ndarray):
        return np_kernel1d(x, kernel_size, kernel=np.amax)
    elif hasattr(x, 'normalized_intervals'):
        return ai_dilate(x, kernel_size)
    else:
        raise TypeError(x)


def min_kernel1d(x, kernel_size):
    """
    Apply a max kernel to x.
    In case of a boolean arrays, this operation is known as dilation.

    >>> a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,])
    >>> _plot(a); _plot(min_kernel1d(a, 3))
    '    ████    '
    '     ██     '
    """
    if isinstance(x, np.ndarray):
        return np_kernel1d(x, kernel_size, kernel=np.amin)
    elif hasattr(x, 'normalized_intervals'):
        return ai_erode(x, kernel_size)
    else:
        raise TypeError(x)


def mean_kernel1d(x, kernel_size):
    """
    Apply a mean kernel to x.

    >>> a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,])
    >>> _plot(a); _plot(mean_kernel1d(a, 3))
    '    ████    '
    '   ▃▅██▅▃   '
    """
    if isinstance(x, np.ndarray):
        return np_kernel1d(x, kernel_size, kernel=np.mean)
    else:
        raise TypeError(x)


def median_kernel1d(x, kernel_size):
    """
    Apply a median kernel to x.

    >>> a = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0,])
    >>> _plot(a); _plot(median_kernel1d(a, 3))
    ' █  ██ ██  █ '
    '    █████    '
    """
    if isinstance(x, np.ndarray):
        return np_kernel1d(x, kernel_size, kernel=np.median)
    else:
        raise TypeError(x)
