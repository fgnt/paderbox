import re
import numpy as np
import collections
import numbers
from numpy.core.einsumfunc import _parse_einsum_input
from dataclasses import dataclass
from paderbox.utils.mapping import Dispatcher


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


def stack_context(X, left_context=0, right_context=0, step_width=1):
    """ Stack TxBxF format with left and right context.

    There is a notebook, which illustrates this feature with many details in
    the example notebooks repository.

    :param X: Data with TxBxF format.
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Stacked features with symmetric padding and head and tail.
    """
    X_stacked = tbf_to_tbchw(
        X,
        left_context=left_context,
        right_context=right_context,
        step_width=step_width
    )[:, :, 0, :].transpose((0, 1, 3, 2))

    T, B, F, W = X_stacked.shape
    X_stacked = X_stacked.reshape(T, B, F * W)

    return X_stacked


def unstack_context(X, mode, left_context=0, right_context=0, step_width=1):
    """ Unstacks stacked features.

    This only works in special cases. Right now, only mode='center'
    is supported. It will return just the center frame and drop the remaining
    parts.

    Other options are related to combining overlapping context frames.

    :param X: Stacked features (or output of your network)
    :param X: mode
    :param left_context: Length of left context.
    :param right_context: Length of right context.
    :param step_width: Step width.
    :return: Data with TxBxF format.
    """

    assert step_width == 1
    context_length = left_context + 1 + right_context
    assert X.shape[2] % context_length == 0
    F = X.shape[2] // context_length

    if mode == 'center':
        return X[:, :, left_context * F:(left_context + 1) * F]
    else:
        NotImplementedError(
            'All other unstack methods are not yet implemented.'
        )


def add_context(data, left_context=0, right_context=0, step=1,
                cnn_features=False, deltas_as_channel=False,
                num_deltas=2, sequence_output=True):
    if cnn_features:
        data = tbf_to_tbchw(data, left_context, right_context, step,
                            pad_mode='constant',
                            pad_kwargs=dict(constant_values=(0,)))
        if deltas_as_channel:
            feature_size = data.shape[3] // (1 + num_deltas)
            data = np.concatenate(
                [data[:, :, :, i * feature_size:(i + 1) * feature_size, :]
                 for i in range(1 + num_deltas)], axis=2)
    else:
        data = stack_context(data, left_context=left_context,
                             right_context=right_context, step_width=step)
        if not sequence_output:
            data = np.concatenate(
                [data[:, i, ...].reshape((-1, data.shape[-1])) for
                 i in range(data.shape[1])], axis=0)
    return data


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
