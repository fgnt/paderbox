import numpy as np
import collections
import numbers


def to_ndarray(data, copy=True):
    if copy:
        cp = lambda x: np.copy(x)
    else:
        cp = lambda x: x
    if str(type(data)) == "<class 'chainer.variable.Variable'>":
        return cp(data.num)
    elif isinstance(data, np.ndarray):
        return cp(data)
    elif isinstance(data, numbers.Number):
        return data
    elif isinstance(data, collections.Iterable):
        return np.asarray(data)
    else:
        raise ValueError('Unknown type of data {}. Cannot add to list'
                         .format(type(data)))


def labels_to_one_hot(
        labels: np.ndarray, categories: int, axis: int = 0,
        keepdims=False, dtype=np.bool
):
    """ Translates an arbitrary ndarray with labels to one hot coded array.

    Args:
        labels: Array with any shape and integer labels.
        categories: Maximum integer label larger or equal to maximum of the
            labels ndarray.
        axis: Axis along which the one-hot vector will be aligned.
        keepdims:
            If keepdims is True, this function behaves similar to
            numpy.concatenate(). It will expand the provided axis.
            If keepdims is False, it will create a new axis along which the
            one-hot vector will be placed.
        dtype: Provides the dtype of the output one-hot mask.

    Returns:
        One-hot encoding with shape (..., categories, ...).

    """
    if keepdims:
        assert labels.shape[axis] == 1
        result_ndim = labels.ndim
    else:
        result_ndim = labels.ndim + 1

    if axis < 0:
        axis += result_ndim

    shape = labels.shape
    zeros = np.zeros((categories, labels.size), dtype=dtype)
    zeros[labels.ravel(), range(labels.size)] = 1

    zeros = zeros.reshape((categories,) + shape)

    if keepdims:
        zeros = zeros[(slice(None),) * (axis + 1) + (0,)]

    zeros = np.moveaxis(zeros, 0, axis)

    return zeros
