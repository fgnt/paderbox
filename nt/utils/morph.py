import re
import string
import functools
import operator

import numpy as np

MESSAGE = 'source: {}, target: {}, current shape: {}'


def morph(operation, array, **shape_hints):
    """
    Copied and revised version of paderflow/ops/morph

    Rationale: An operation consists of source and target annotation. If a
    dimension is in source but not in target, is has to be a singleton
    dimension. If a dimension is not in source but in target, it will be a new
    singleton dimension. If the order of annotations changes, it will result in
    a transpose. An asterisk in the target annotation will result in a reshape
    to fit the new shape. If you provide shape hints for newly introduced
    shapes it will use a tile operation in that dimension instead of a
    singleton dimension.

    :param operation: Shape annotation of source and target. See test cases.
    :param array: Single tensor to be reshaped and transposed.
    :param shape_hints: If a reshape requires further information to be
        performed, provide the dimension names as keys and the dimension as
        value.
    :return:
    """

    operation = operation.replace(' ', '')
    operation = operation.replace(',', '')
    source, target = map(str.lower, operation.split('->'))

    for d in source:
        assert d in string.ascii_lowercase + '*' + '1'
    for d in target:
        assert d in string.ascii_lowercase + '*' + '1'

    if array.shape is not None:
        assert (
            len(array.shape) == len(source.replace('*', '')) - source.count(
                '*')
        ), MESSAGE.format(source, target, array.shape)

    array = _expanding_reshape(array, source, **shape_hints)
    source = source.replace('*', '')

    array = _remove_undesired_singleton(array, source, target)
    source = ''.join([c for c in source if c in target])

    transpose_target = ''.join([char for char in target if char in source])
    array = _transpose(array, source, transpose_target)
    source = transpose_target

    expand_target = target.replace('*', '')
    array = _expand_desired_singleton(
        array, source, expand_target, **shape_hints
    )
    source = expand_target

    array = _shrinking_reshape(array, source, target)
    return array


def _expanding_reshape(array, source, **shape_hints):
    """

    :param array:
    :param source:
    :param shape_hints:
    :return:
    """
    if '*' not in source:
        return array

    target_shape = []

    for axis, group in enumerate(_get_source_grouping(source)):
        if len(group) == 1:
            target_shape.append(array.shape[axis:axis + 1])
        else:
            shape_wildcard_remaining = True
            for member in group:
                if member in shape_hints:
                    target_shape.append([shape_hints[member]])
                else:
                    if shape_wildcard_remaining:
                        shape_wildcard_remaining = False
                        target_shape.append([-1])
                    else:
                        raise ValueError('Not enough shape hints provided.')

    target_shape = np.concatenate(target_shape, 0)
    array = array.reshape(target_shape)
    return array


def _remove_undesired_singleton(array, source, target):
    """
    Squeeze away dimensions which do not appear in target shape.
    """
    assert '*' not in source, MESSAGE.format(source, target, array.shape)
    if set(source) == set(target):
        return array

    squeezer = [slice(0, None) if d in target else 0 for d in source]
    array = array[squeezer]

    return array


def _transpose(array, source, target):
    assert '*' not in source, MESSAGE.format(source, target, array.shape)
    assert '*' not in target, MESSAGE.format(source, target, array.shape)
    assert (
        set(source) == set(target)
    ), MESSAGE.format(source, target, array.shape)

    if source == target:
        return array

    source = {t: i for i, t in enumerate(source)}
    perm = [source[s] for s in target]
    return np.transpose(array, perm)


def _expand_desired_singleton(array, source, target, **shape_hints):
    assert '*' not in source, MESSAGE.format(source, target, array.shape)
    assert '*' not in target, MESSAGE.format(source, target, array.shape)

    if source == target:
        return array

    expander = [slice(0, None) if d in source else None for d in target]
    array = array[expander]

    new = set(target) - set(source)

    def get_tiling(ax):
        if ax in new and ax in shape_hints:
            return shape_hints[ax]
        else:
            return 1

    array = np.tile(
        array,
        [get_tiling(ax) for ax in target]
    )

    return array


def _shrinking_reshape(array, source, target):
    """
    Folds two or more dimensions if `*` in target shape.
    """

    assert '*' not in source, MESSAGE.format(source, target, array.shape)
    assert '*' != target[0], MESSAGE.format(source, target, array.shape)
    assert '*' != target[-1], MESSAGE.format(source, target, array.shape)
    assert (
        source == target.replace('*', '')
    ), MESSAGE.format(source, target, array.shape)
    if source == target:
        return array

    for begin, end in _get_target_grouping(target):
        array = reshape(
            array, [-1], begin_axis=begin, end_axis=end)
    return array


def _get_target_grouping(target):
    """
    Get axis indices for each target group for a shrinking reshape op.

    Gets axis as numeric.

    Each group tuple (begin, end) describes the following reshape
    shape = source.shape
    shape[begin:end] = [-1]
    target.shape == shape

    Note: The order is important, because the execution of the reshape is
    sequential and not concurrent.
    """
    source = target.replace('*', '')
    assert target[0] != '*', MESSAGE.format(source, target, '')
    assert target[-1] != '*', MESSAGE.format(source, target, '')

    mapping = {t: i for i, t in enumerate(source.replace('*', ''))}

    #target = re.sub('\*.\*', '*', target)  # Example: ...*b*... -> ...*...

    result = []
    for t, prev_t in zip(target[1:], target[:-1]):
        if t == '*':
            idx = mapping[prev_t]
            result.append([idx, idx + 2])

    return list(reversed(result))


def _get_source_grouping(source):
    """
    Used for expanding reshape op. Gets axis as alphanumeric.
    """

    source = ' '.join(source)
    source = source.replace(' * ', '*')
    groups = source.split()
    groups = [group.split('*') for group in groups]
    return groups


def reshape(
        array,
        shape,
        begin_axis=None,
        end_axis=None
):
    """
    Inspired from cntk.reshape to allow begin_axis and end_axis

    :param array:
    :param shape:
    :param begin_axis:
    :param end_axis:
    :return:
    """

    if begin_axis is None and end_axis is None:
        return array.reshape(shape)

    if shape == [-1]:
        tmp = array.shape[begin_axis:end_axis]
        if tmp is not None:
            tmp = functools.reduce(operator.mul, tmp, (1))
        if str(tmp) == '?' or tmp is None:
            shape = [-1]
        else:
            shape = [tmp]
        to_concat = [shape]
    else:
        to_concat = [shape]
    if begin_axis is not None:
        bs = array.shape[:begin_axis]
        to_concat.insert(0, bs)
    if end_axis is not None:
        es = array.shape[end_axis:]
        to_concat.append(es)

    try:
        array_shape = np.concatenate(to_concat, 0)
    except TypeError as e:
        raise TypeError(to_concat) from e

    res = array.reshape(array_shape.astype(np.int16))
    return res


def ravel(array, begin_axis=None, end_axis=None):
    return reshape(array, [-1], begin_axis=begin_axis, end_axis=end_axis)
