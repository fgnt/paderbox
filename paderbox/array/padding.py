import numpy as np
from dataclasses import dataclass

__all__ = [
    'pad_to',
    'pad_axis',
    'roll_zeropad',
    'Cutter',
]


def pad_to(array, to, constant_value=0):
    """ One dimensional padding with zeros to the size of the target array

    :param array: Input array which will be part of the result
    :param to: Target array. Its size will be used to determine the size of the
        return array.
    :return: Padded array
    """
    array = np.array(array)
    result = constant_value * np.ones((len(to),), dtype=array.dtype)
    result[:array.shape[0]] = array
    return result


def pad_axis(array, pad_width, *, axis, mode='constant', **pad_kwargs):
    """ Wrapper around np.pad to support the axis argument.
    This function has mode='constant' as default.

    >>> pad_axis(np.ones([3, 4]), 1, axis=0)
    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]])
    >>> pad_axis(np.ones([3, 4]), 1, axis=1)
    array([[0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.]])
    >>> pad_axis(np.ones([3, 4]), (0, 1), axis=1)
    array([[1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.]])
    >>> pad_axis(np.ones([3, 4]), (1, 0), axis=1)
    array([[0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.]])

    Since np.pad has no axis argument the behaviour for
    isinstance(pad_width, int) is rarely the desired behaviour:

    >>> np.pad(np.ones([3, 4]), 1, mode='constant')
    array([[0., 0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0., 0.]])

    Here the corresponding np.pad calls for above examples:

    >>> np.pad(np.ones([3, 4]), ((1,), (0,)), mode='constant')
    array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0,), (1,)), mode='constant')
    array([[0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0, 0), (0, 1)), mode='constant')
    array([[1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.],
           [1., 1., 1., 1., 0.]])
    >>> np.pad(np.ones([3, 4]), ((0, 0), (1, 0)), mode='constant')
    array([[0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.],
           [0., 1., 1., 1., 1.]])


    """
    array = np.asarray(array)

    npad = np.zeros([array.ndim, 2], dtype=np.int)
    npad[axis, :] = pad_width
    return np.pad(array, pad_width=npad, mode=mode, **pad_kwargs)


# http://stackoverflow.com/a/3153267
def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros),
                             axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


@dataclass
class Cutter:
    """
    Implements cut and expand for low_cut and high_cut. Often interesting when
    you want to avoid processing of some frequencies when beamforming.

    Why do we enforce negative upper end: Positive values can be confusing. You
    may want to cut `n` values or want to keep up to `n`.

    To implement similar behaviour in Torch, you may use `torch.narrow()`:
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.narrow

    >>> c = Cutter(1, -2)
    >>> array = np.array([[1, 2, 3, 4]])
    >>> c.cut(array, axis=1)
    array([[2]])

    >>> c.expand(c.cut(array, axis=1), axis=1)
    array([[0, 2, 0, 0]])

    >>> c.overwrite(array, axis=1)
    array([[0, 2, 0, 0]])

    >>> c = Cutter(0, None)
    >>> c.cut(array, axis=1)
    array([[1, 2, 3, 4]])
    """
    low_cut: int
    high_cut: int

    def __post_init__(self):
        assert self.low_cut >= 0, 'Zero or positive'
        assert self.high_cut is None or self.high_cut <= 0, 'None or negative'

    def cut(self, array, *, axis):
        """Cuts start and end."""
        assert isinstance(axis, int), axis
        trimmer = [slice(None)] * array.ndim
        trimmer[axis] = slice(self.low_cut, self.high_cut)
        return array[tuple(trimmer)]

    def expand(self, array, *, axis):
        """Pads to reverse the cut."""
        assert isinstance(axis, int), axis
        if self.high_cut is None:
            upper_pad = 0
        else:
            upper_pad = -self.high_cut
        return pad_axis(array, (self.low_cut, upper_pad), axis=axis)

    def overwrite(self, array, *, axis):
        """Returns a copy with start end end filled with zeros."""
        return self.expand(self.cut(array, axis=axis), axis=axis)
