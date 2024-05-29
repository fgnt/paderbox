import bisect
import copy
import dataclasses
from dataclasses import dataclass, field

from typing import Any, Optional, Type

import textwrap

from paderbox.array.interval.core import ArrayInterval
import numpy as np
import paderbox as pb
from paderbox.array.interval.util import cy_parse_item, cy_invert_intervals

try:
    import torch

    array_types = (np.ndarray, torch.Tensor)

    # Operations whose neutral element is 0
    shortcut_operations = (
        np.add, np.subtract, torch.Tensor.add_,
        torch.Tensor.add,
        torch.Tensor.sub_, torch.Tensor.sub
    )
except ImportError:
    array_types = (np.ndarray,)
    shortcut_operations = (np.add, np.subtract)


def _normalize_shape(shape):
    """Performs a few checks on the shape and normalizes it to a tuple"""
    allowed_types = (int, np.integer)

    if shape is None:
        # As of now, we only support fixed shapes and an unknown time
        # dimension. This can be changed
        raise TypeError(f'None shape is not supported')

    if isinstance(shape, allowed_types):
        return shape,

    if not isinstance(shape, (tuple, list)):
        raise TypeError(f'Invalid shape {shape} of type {type(shape)}')

    if (not all(isinstance(s, allowed_types) for s in shape[:-1])
        and not (shape[-1] is None or isinstance(shape[-1], allowed_types))
    ):
        raise TypeError(
            f'Invalid shape {shape} with elements of type {type(shape[0])}'
        )

    return tuple(shape)


def _parse_item(item, shape):
    if shape[-1] is None:
        start = item.start
        if start is None:
            start = 0
        assert start >= 0, item
        assert item.stop is None or item.stop >= 0, item

        return start, item.stop
    else:
        return cy_parse_item(item, shape)


def _shape_for_item(shape, item):
    """Gets the shape that would result when indexing an array with shape
    `shape` with `item`."""
    if shape == ():
        return ()
    if shape[-1] is None:
        return np.broadcast_to(np.ones(1), shape[:-1])[item[:-1]].shape + (None,)
    else:
        return np.broadcast_to(np.ones(1), shape)[item].shape


def _normalize_item(item, ndim):
    """
    >>> def print_item(item):
    ...     print('[' + ', '.join([(f'{i.start if i.start is not None else ""}:{i.stop if i.stop is not None else ""}' if isinstance(i, slice) else str(i)) for i in item]) + ']')
    >>> print_item((0, slice(None), slice(1), slice(5, 10)))
    [0, :, :1, 5:10]
    >>> print_item(_normalize_item((), 3))
    [:, :, :]
    >>> print_item(_normalize_item((0,), 3))
    [0, :, :]
    >>> print_item(_normalize_item(0, 3))
    [0, :, :]
    >>> print_item(_normalize_item(slice(None), 3))
    [:, :, :]
    >>> print_item(_normalize_item(slice(10), 3))
    [:10, :, :]
    >>> print_item(_normalize_item(..., 3))
    [:, :, :]
    >>> print_item(_normalize_item((0, ...), 3))
    [0, :, :]
    >>> print_item(_normalize_item((..., 0), 3))
    [:, :, 0]
    >>> print_item(_normalize_item((1, 2, 3, ...), 3))
    [1, 2, 3]
    >>> print_item(_normalize_item((..., 1, 2, 3, ...), 3))
    Traceback (most recent call last):
     ...
    IndexError: an index can only have a single ellipsis ('...')
    >>> print_item(_normalize_item((0, 1, 2, 3, ...), 3))
    Traceback (most recent call last):
      ...
    IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed
    """
    if isinstance(item, list):
        raise NotImplementedError()

    if not isinstance(item, tuple):
        item = (item,)
    item = list(item)
    if ... in item:
        assert ndim is not None, ndim
        idx = item.index(...)
        item[idx:idx + 1] = [slice(None)] * (ndim - len(item) + 1)
    elif ndim is not None:
        item = item + [slice(None)] * (ndim - len(item))
    if ... in item:
        raise IndexError('an index can only have a single ellipsis (\'...\')')
    if ndim is not None and len(item) > ndim:
        raise IndexError(
            f'too many indices for array: array is {ndim}-dimensional, '
            f'but {len(item)} were indexed'
        )
    return item


def _dtype_from_value(value):
    try:
        return value.dtype
    except AttributeError:
        # This is a fallback for builtin types
        return np.dtype(type(value))


def _get_pad_value(dtype, pad_value, device=None):
    if 'torch' in str(dtype):
        import torch
        return torch.full(
            size=(1,), fill_value=pad_value, dtype=dtype, device=device
        )
    else:
        return np.full(shape=(1,), fill_value=pad_value, dtype=dtype)


def _pad_value_like(array, pad_value):
    """Constructs a pad value with the same dtype and device as `array`."""
    device = getattr(array, 'device', None)  # Works for numpy and torch
    return _get_pad_value(_dtype_from_value(array), pad_value, device=device)


def full(shape, pad_value, dtype=np.float32):
    """
    Creates an empty `SparseArray` with a pad value of `pad_value`.

    Args:
        shape: The shape of the array. All dimensions must be specified except
            for the last dimension, which can be `None`. In that case, the
            time dimension is dynamic and grows to fit any segments added. The
            shape can be persisted by calling `SparseArray.persist_shape`.
        pad_value: The value to pad with where no data is set
        dtype: Datatype of the created array

    >>> full(10, 5)
    SparseArray(pad_value=5.0, shape=(10,))
    >>> full(10, 5).as_contiguous()
    array([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.], dtype=float32)
    >>> full(10, 5, dtype=torch.int32).as_contiguous()
    tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=torch.int32)
    """
    return SparseArray(shape=shape, _pad_value=_get_pad_value(dtype, pad_value))


def zeros(shape, dtype=np.float32):
    """
    Creates an empty `SparseArray` with a pad value of 0.

    Args:
        shape: The shape of the array. All dimensions must be specified except
            for the last dimension, which can be `None`. In that case, the
            time dimension is dynamic and grows to fit any segments added. The
            shape can be persisted by calling `SparseArray.persist_shape`.
        dtype: Datatype of the created array

    >>> zeros(10)
    SparseArray(shape=(10,))
    >>> zeros(10).as_contiguous()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> zeros(10, dtype=np.float64).as_contiguous()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> zeros(10, dtype=torch.float32).as_contiguous()
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    >>> zeros(10, dtype=torch.float32).device
    device(type='cpu')
    """
    return full(shape, dtype=dtype, pad_value=0)


def from_array_interval(
        array_interval: ArrayInterval,
        dtype=bool,
        *,
        true_value=1,
        false_value=0
) -> 'SparseArray':
    """
    Constructs a `SparseArray` from an `ArrayInterval`.

    >>> from paderbox.array import interval
    >>> ai = interval.zeros(10)
    >>> from_array_interval(ai)
    SparseArray(shape=(10,))
    >>> ai[:5] = 1; ai[7:] = 1
    >>> from_array_interval(ai).as_contiguous()
    array([ True,  True,  True,  True,  True, False, False,  True,  True,
            True])
    >>> ai = interval.ones(5)
    >>> from_array_interval(ai).as_contiguous()
    array([ True,  True,  True,  True,  True])
    >>> ai[:2] = False
    >>> from_array_interval(ai).as_contiguous()
    array([False, False,  True,  True,  True])
    """
    if array_interval.inverse_mode:
        array = full(array_interval.shape, pad_value=true_value, dtype=dtype)
        interval_value = false_value
    else:
        array = zeros(array_interval.shape, dtype=dtype)
        interval_value = true_value
    for (start, stop) in array_interval.normalized_intervals:
        array[..., start:stop] = interval_value
    return array


def _check_shape(shape1, shape2):
    if shape1 != shape2:
        raise ValueError(f'Shape mismatch: {shape1} {shape2}')


# Use dataclass for recursive to_numpy
@dataclass(frozen=True)
class _SparseSegment:
    onset: int
    array: array_types

    @classmethod
    def from_array(cls, array, onset, target_length=None):
        offset = onset + array.shape[-1]
        if onset < 0:
            array = array[..., -onset:]
            onset = 0
        if target_length is not None and offset > target_length:
            array = array[..., :target_length - onset]
        return cls(onset, array)

    @property
    def offset(self):
        return self.onset + self.array.shape[-1]

    @property
    def segment_length(self):
        return self.array.shape[-1]

    @property
    def leading_shape(self):
        return self.array.shape[:-1]

    def __array__(self):
        """
        >>> seg = _SparseSegment(10, np.zeros(10))
        >>> np.array(seg)
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        if not isinstance(self.array, np.ndarray):
            raise NotImplementedError(
                f'__array__ is not implemented for torch '
                f'{self.__class__.__name__}'
            )
        return self.array


@dataclass
class SparseArray:
    """
    A sparse array implementation using intervals to define non-zero regions.

    >>> a = SparseArray(shape=(20, ))
    >>> a
    SparseArray(shape=(20,))
    >>> a[5:10] = np.ones(5, dtype=np.float64)
    >>> a
    SparseArray(_SparseSegment(onset=5, array=array([1., 1., 1., 1., 1.])), shape=(20,))
    >>> np.asarray(a)
    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0.])
    >>> a[15:20] = 2
    >>> a
    SparseArray(_SparseSegment(onset=5, array=array([1., 1., 1., 1., 1.])), _SparseSegment(onset=15, array=array([2., 2., 2., 2., 2.])), shape=(20,))
    >>> np.asarray(a)
    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 2., 2.,
           2., 2., 2.])

    Overlapping intervals are not allowed
    >>> a[14:16] = 3
    Traceback (most recent call last):
      ...
    ValueError: Overlap detected between
      _SparseSegment(onset=15, array=array([2., 2., 2., 2., 2.]))
    and newly added
      _SparseSegment(onset=14, array=array([3., 3.]))


    dtype is inferred from segments:
    >>> a.dtype
    dtype('float64')
    >>> a.pad_value, type(a.pad_value)
    (0.0, <class 'numpy.float64'>)

    Can't add segments with differing dtypes
    >>> a[:3] = np.ones(3, dtype=np.float32)
    Traceback (most recent call last):
      ...
    TypeError: Type mismatch: SparseArray has dtype float64, but assigned array has dtype float32

    Adding multiple SparseArray returns a SparseArray
    >>> b = SparseArray(a.shape)
    >>> b[10:15] = 8.0
    >>> b.dtype
    dtype('float64')
    >>> a + b
    SparseArray(_SparseSegment(onset=5, array=array([1., 1., 1., 1., 1.])), _SparseSegment(onset=10, array=array([8., 8., 8., 8., 8.])), _SparseSegment(onset=15, array=array([2., 2., 2., 2., 2.])), shape=(20,))
    >>> np.asarray(a + b)
    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 8., 8., 8., 8., 8., 2., 2.,
           2., 2., 2.])

    Arithmetic works with numpy arrays
    >>> np.ones(20, dtype=np.float64) * a
    array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 2., 2.,
           2., 2., 2.])
    >>> np.ones(20, dtype=np.float64) + a
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 3., 3.,
           3., 3., 3.])
    >>> a + np.ones(20, dtype=np.float64)
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 3., 3.,
           3., 3., 3.])
    >>> c = np.ones(20, dtype=np.float64)
    >>> c += a
    >>> c
    array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 3., 3.,
           3., 3., 3.])

    Test if pt.data.batch.example_to_device converts correctly. Commented out
    because these tests don't work without padertorch
    # >>> import padertorch as pt
    # >>> a = pt.data.batch.example_to_device(a)
    # >>> a
    # SparseArray(_SparseSegment(onset=5, array=tensor([1, 1, 1, 1, 1])), _SparseSegment(onset=15, array=tensor([2, 2, 2, 2, 2])), shape=(20,))
    # >>> a.dtype, a.device, a.is_torch
    # (torch.int64, device(type='cpu'), True)
    # >>> a = pt.data.batch.example_to_numpy(a)
    # >>> a
    # SparseArray(_SparseSegment(onset=5, array=array([1, 1, 1, 1, 1])), _SparseSegment(onset=15, array=array([2, 2, 2, 2, 2])), shape=(20,))
    # >>> a.dtype, a.device, a.is_torch
    # (dtype('int64'), 'numpy', False)
    # >>> a = pt.data.batch.example_to_device(zeros(10))
    # >>> a
    # SparseArray(shape=(10,))
    # >>> a.dtype, a.device, a.is_torch
    # (torch.float32, device(type='cpu'), True)
    """
    # Public constructor interface
    shape: tuple

    # Private constructor arguments: Don't use _segments or _pad_value directly.
    # This is required for pt.data.batch.example_to_device to work
    _segments: list = field(default_factory=list)
    # The _pad_value property tracks the pad value and the type, dtype and
    # possibly device of the data. It is a one-dimensional numpy array or a
    # torch tensor with a single element so that example_to_device transfers
    # it correctly between torch and numpy.
    # If _pad_value is None, the SparseArray doesn't have a dtype and it will be
    # defined by the first added segment.
    _pad_value: Any = None

    def __post_init__(self):
        self.shape = _normalize_shape(self.shape)

    @property
    def dtype(self):
        """
        >>> SparseArray(10).dtype
        >>> zeros(10).dtype
        dtype('float32')
        >>> a = SparseArray(10)
        >>> a[:5] = np.arange(5, dtype=np.float64)
        >>> a.dtype
        dtype('float64')
        >>> a = SparseArray(10)
        >>> a[:5] = torch.arange(5)
        >>> a.dtype
        torch.int64
        """
        if self._pad_value is None:
            return None
        return self._pad_value.dtype

    @property
    def device(self):
        """
        The device of the `SparseArray`.

        Returns `'numpy'` as the device if the contents are numpy arrays.

        >>> a = SparseArray(10)
        >>> a.device  # is None
        >>> a[:5] = 1
        >>> a.device
        'numpy'
        >>> zeros(10).device
        'numpy'
        >>> zeros(10, dtype=torch.float32).device
        device(type='cpu')
        """
        if self._pad_value is None:
            return None
        if isinstance(self._pad_value.dtype, np.dtype):
            return 'numpy'
        return self._pad_value.device

    @property
    def is_torch(self):
        if self._pad_value is None:
            return None
        return 'torch' in str(self._pad_value.dtype)

    @property
    def pad_value(self):
        """
        >>> a = SparseArray(10)
        >>> a.pad_value   # is None
        >>> a[:5] = np.arange(5)
        >>> a.pad_value
        0
        >>> zeros(10).pad_value
        0.0
        >>> full(10, 42).pad_value
        42.0
        """
        # _pad_value must be an array, otherwise pt.data.batch.example_to_device
        # can't convert between numpy and torch correctly. The actual pad_value
        # should be a scalar
        if self._pad_value is None:
            return None
        return self._pad_value[0]

    @property
    def _leading_shape(self):
        return self.shape[:-1]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def _time_length(self):
        return self.shape[-1]

    @property
    def _estimated_time_length(self):
        if self._time_length is None:
            if len(self._segments) == 0:
                raise RuntimeError(f'Can\'t determine shape of empty SparseArray')
            return max(0, max(s.offset for s in self._segments))
        return self._time_length

    @classmethod
    def from_array_and_onset(cls, array, onset, shape=None, pad_value=0.):
        """
        Creates a `SparseArray` from a numpy array or torch tensor with a given
        `onset` and `shape`. The given `array` is cut so that it lies within
        the boundaries defined by `shape`, e.g., a negative onset will become
        0 and the part of `array` that would lie outside the `SparseArray` would
        be cut.

        >>> SparseArray.from_array_and_onset(np.ones(5), 2, 10).as_contiguous()
        array([0., 0., 1., 1., 1., 1., 1., 0., 0., 0.])
        >>> SparseArray.from_array_and_onset(np.ones(5), -3, 10)
        SparseArray(_SparseSegment(onset=0, array=array([1., 1.])), shape=(10,))
        >>> SparseArray.from_array_and_onset(np.ones(5), -3, 10).as_contiguous()
        array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> SparseArray.from_array_and_onset(np.ones(5), 7, 10)
        SparseArray(_SparseSegment(onset=7, array=array([1., 1., 1.])), shape=(10,))
        >>> SparseArray.from_array_and_onset(np.ones(5), 7).as_contiguous()
        array([0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
        """
        return cls.from_arrays_and_onsets([array], [onset], shape, pad_value)

    @staticmethod
    def from_arrays_and_onsets(arrays, onsets, shape=None, pad_value=0.):
        """
        >>> SparseArray.from_arrays_and_onsets([np.ones(2), np.ones(1)], [0, 3]).as_contiguous()
        array([1., 1., 0., 1.])
        >>> SparseArray.from_arrays_and_onsets([np.ones(2), np.ones(1)], [0, 3], (10,)).as_contiguous()
        array([1., 1., 0., 1., 0., 0., 0., 0., 0., 0.])
        """
        if shape is None:
            # Infer leading shape from arrays, but we can't infer the shape along
            # time dimension
            shape = arrays[0].shape[:-1] + (None,)

        out = SparseArray(shape=shape, _pad_value=_pad_value_like(arrays[0], pad_value))

        if len(arrays) != len(onsets):
            raise ValueError(
                f'Number of arrays and onsets must match! '
                f'len(arrays)={len(arrays)} len(onsets)={len(onsets)}'
            )

        for array, onset in zip(arrays, onsets):
            out._add_segment(_SparseSegment.from_array(array, onset, out.shape[-1]))

        return out

    def as_contiguous(self, dtype=None, infer_shape=True):
        """
        Converts the `SparseArray` to a numpy array or torch tensor, depending
        on the data in the `SparseArray`.

        Args:
            dtype: The data type of the returned array
            infer_shape: If set to true and the time dimension of the
                `SparseArray` is `None`, the shape is inferred from the
                contained data

        >>> zeros((None,)).as_contiguous()
        array([], dtype=float32)
        >>> zeros((2, None)).as_contiguous()
        array([], shape=(2, 0), dtype=float32)
        >>> a = zeros((None,))
        >>> a[:3] = 1
        >>> a.as_contiguous()
        array([1., 1., 1.], dtype=float32)
        """
        shape = self.shape
        if shape[-1] is None:
            if not infer_shape:
                raise ValueError(
                    f'Cannot construct numpy array from unknown shape. '
                    f'Did you forget to call `persist_shape` or to set '
                    f'`infer_shape=True`?'
                )
            if len(self._segments) == 0:
                shape = shape[:-1] + (0,)
            else:
                shape = shape[:-1] + (self._segments[-1].offset,)
        arr = self._new_full(dtype=dtype, shape=shape)
        for segment in self._segments:
            arr[..., segment.onset:segment.offset] = segment.array
        return arr

    @property
    def interval(self) -> ArrayInterval:
        ai = pb.array.interval.zeros(self._time_length)
        s: _SparseSegment
        for s in self._segments:
            ai[s.onset:s.offset] = True
        return ai

    def _new_full(self, *, shape=None, dtype=None, fill_value=None):
        """
        >>> np.array(zeros(10))
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
        >>> np.array(SparseArray(10))
        Traceback (most recent call last):
        ...
        RuntimeError: You cannot convert SparseArray to numpy,
        when self.pad_value is None`.
        Did you use `pb.array.sparse.SparseArray(shape)` with
        lazy dtype init instead of `pb.array.sparse.zeros(shape, dtype)`
        to create this `SparseArray`?
        """
        if fill_value is None:
            fill_value = self.pad_value
            if fill_value is None:
                lib = ['numpy', 'torch'][bool(self.is_torch)]
                raise RuntimeError(
                    f'You cannot convert {self.__class__.__name__} to {lib},\n'
                    f'when self.pad_value is None`.\n'
                    f'Did you use `pb.array.sparse.SparseArray(shape)` with\n'
                    f'lazy dtype init instead of '
                    f'`pb.array.sparse.zeros(shape, dtype)`\n'
                    f'to create this `{self.__class__.__name__}`?'
                )
        if shape is None:
            shape = self.shape
        if shape == ():
            return fill_value
        elif shape is None or shape[-1] is None:
            raise TypeError(
                'Shape is None or contains None, can\'t build an '
                'array from it.'
            )
        if self.is_torch:
            return self.pad_value.new_full(
                size=shape, fill_value=float(fill_value), dtype=dtype,
            )
        else:
            return np.full_like(self.pad_value, fill_value=fill_value,
                                shape=shape, dtype=dtype)

    def persist_shape(self):
        """
        Persists the current shape of
        """
        if self._time_length is not None:
            return self.shape
        self.shape = self._leading_shape + (self._estimated_time_length,)
        return self.shape

    def _add_segment(self, segment: _SparseSegment):
        if self._pad_value is not None:
            # We can't mix numpy with torch
            if self.is_torch == isinstance(segment.array, np.ndarray):
                raise TypeError(
                    f'{type(segment.array)} is incompatible to {type(self._pad_value)}'
                )

            # dtype has to match for now. Supporting a mix of compatible dtypes is
            # difficult to implement
            if self.dtype != segment.array.dtype:
                raise TypeError(
                    f'Type mismatch: SparseArray has dtype {self.dtype}, but '
                    f'assigned array has dtype {segment.array.dtype}'
                )

            # The leading dimensions must be equal
            if self._leading_shape != segment.leading_shape:
                raise ValueError(
                    f'Shape mismatch: SparseArray has leading shape {self._leading_shape}, '
                    f'but assigned array has leading '
                    f'shape {segment.array.leading_shape}.'
                )

            # If we have a torch tensor, also check the device. Having parts of the
            # data distributed across devices is not desired
            if self.is_torch:
                if self.device != segment.array.device:
                    raise ValueError(
                        f'Mixed devices are not supported. Sparse Arrray on '
                        f'device {self.device}, but assigned array is on '
                        f'device {segment.array.device}'
                    )

        if segment.offset < 0 or self._time_length is not None and segment.onset >= self._time_length:
            # Ignore setting anything that is outside of the boundaries (numpy
            # behavior)
            return

        # Get insert position. Keep the arrays sorted to speed up checks for
        # overlaps and getitem
        position = bisect.bisect_right(
            [s.onset for s in self._segments],
            segment.onset
        )

        # Check for overlaps with neighboring segments
        if (
                position > 0
                and self._segments[position - 1].offset > segment.onset
        ):
            raise ValueError(
                f'Overlap detected between\n'
                f'{textwrap.indent(repr(self._segments[position - 1]), "  ")}\n'
                f'and newly added\n'
                f'{textwrap.indent(repr(segment), "  ")}'
            )
        if (
                position < len(self._segments)
                and self._segments[position].onset < segment.offset
        ):
            raise ValueError(
                f'Overlap detected between\n'
                f'{textwrap.indent(repr(self._segments[position]), "  ")}\n'
                f'and newly added\n'
                f'{textwrap.indent(repr(segment), "  ")}'
            )

        # Update the pad value from the array if the SparseArray doesn't have
        # a pad value yet
        if not self._segments:
            if self.pad_value is None:
                self._pad_value = _pad_value_like(segment.array, 0)
        self._segments.insert(position, segment)

    def __copy__(self):
        # _segments has to be a new list to support __add__, ...
        return dataclasses.replace(self, _segments=list(self._segments))

    def __array__(self, dtype=None):
        if self.is_torch:
            raise NotImplementedError(
                '__array__ is not implemented for torch SparseArrays'
            )
        return self.as_contiguous(dtype, infer_shape=False)

    def __repr__(self):
        if self._segments:
            content = ', '.join(map(str, self._segments))
            content = content + ', '
        else:
            content = ''
        if self.pad_value is not None and self.pad_value != 0:
            p = f'pad_value={self.pad_value}, '
        else:
            p = ''
        return f'{self.__class__.__name__}({content}{p}shape={self.shape})'

    def _repr_pretty_(self, p, cycle):
        """
        >>> pb.utils.pretty.pprint(zeros(10))
        SparseArray(shape=(10,))
        >>> a = zeros(10)
        >>> a[:5] = 1
        >>> a[7:] = 2
        >>> pb.utils.pretty.pprint(a)
        SparseArray(_SparseSegment(onset=0,
                                   array=array([1., 1., 1., 1., 1.], dtype=float32)),
                    _SparseSegment(onset=7, array=array([2., 2., 2.], dtype=float32)),
                    shape=(10,))
        >>> a._pad_value = _get_pad_value(a.dtype, -1)
        >>> pb.utils.pretty.pprint(a)
        SparseArray(_SparseSegment(onset=0,
                                   array=array([1., 1., 1., 1., 1.], dtype=float32)),
                    _SparseSegment(onset=7, array=array([2., 2., 2.], dtype=float32)),
                    shape=(10,), pad_value=-1.0)
        """
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            name = self.__class__.__name__
            pre, post = f'{name}(', ')'
            with p.group(len(pre), pre, post):
                for idx, m in enumerate(self._segments):
                    p.pretty(m)
                    p.text(',')
                    p.breakable()
                p.text(f'shape={self.shape}')
                if self.pad_value != 0:
                    p.text(f', pad_value={self.pad_value}')

    def __setitem__(self, item, value):
        """
        >>> a = SparseArray(10)
        >>> a[:3] = 1
        >>> a[3:5] = np.arange(2) + 1
        >>> a[6:8] = np.ones(2, dtype=int)
        >>> np.asarray(a)
        array([1, 1, 1, 1, 2, 0, 1, 1, 0, 0])

        >>> a = SparseArray(10)
        >>> a[:100] = 1
        >>> a
        SparseArray(_SparseSegment(onset=0, array=array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])), shape=(10,))

        # Multi-dimensional
        >>> a = SparseArray((2, 10))
        >>> a[..., :5] = 1
        >>> a[..., 7:] = np.arange(6).reshape((2, 3))
        >>> np.asarray(a)
        array([[1, 1, 1, 1, 1, 0, 0, 0, 1, 2],
               [1, 1, 1, 1, 1, 0, 0, 3, 4, 5]])
        >>> a[0, 5:] = 2
        Traceback (most recent call last):
          ...
        IndexError: SparseArray.__setitem__ only supports slices for indexing, not [0, slice(5, None, None)]

        # Test shape=None
        >>> a = zeros((None,))
        >>> a[:, :2] = 1
        Traceback (most recent call last):
          ...
        IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        >>> a[:2] = 1
        >>> a[3:] = 2
        Traceback (most recent call last):
          ...
        IndexError: Could not determine size of slice ([slice(3, None, None)])
        >>> a
        SparseArray(_SparseSegment(onset=0, array=array([1., 1.], dtype=float32)), shape=(None,))
        >>> a = zeros((1, None))
        >>> a[:, :2] = 1

        >>> a.as_contiguous()
        array([[1., 1.]], dtype=float32)

        >>> a = zeros((2, None))
        >>> a[:, :5] = np.ones((2, 5), dtype=np.float32)
        >>> a
        SparseArray(_SparseSegment(onset=0, array=array([[1., 1., 1., 1., 1.],
               [1., 1., 1., 1., 1.]], dtype=float32)), shape=(2, None))

        # Test some exceptions
        >>> a = SparseArray(shape=10)
        >>> a[:5] = np.ones(3)
        Traceback (most recent call last):
         ...
        ValueError: Could not assign array with shape (3,) to SparseArray slice with size 5
        >>> a[:5] = np.ones((1, 5))
        Traceback (most recent call last):
         ...
        ValueError: Shape mismatch: SparseArray has shape (10,), but assigned array has shape (1, 5).
        >>> a[:100] = np.ones(100)
        Traceback (most recent call last):
         ...
        ValueError: Could not assign array with shape (100,) to SparseArray slice with size 10
        >>> a = SparseArray((5, 10))
        >>> a[:4, :] = 1
        Traceback (most recent call last):
          ...
        IndexError: Cannot partially set values in the leading dimensions ([slice(None, 4, None)] at dimension 0)
        >>> a[0, :] = 1
        Traceback (most recent call last):
          ...
        IndexError: SparseArray.__setitem__ only supports slices for indexing, not [0, slice(None, None, None)]
        >>> a[:, 0] = 1
        Traceback (most recent call last):
          ...
        IndexError: SparseArray.__setitem__ only supports slices for indexing, not [slice(None, None, None), 0]
        """
        item = _normalize_item(item, self.ndim)

        length = len(item)
        for idx, i in enumerate(item):
            # All items must be slices because we can't set scalar values
            if not isinstance(i, slice):
                raise IndexError(
                    f'{self.__class__.__name__}.__setitem__ only supports '
                    f'slices for indexing, not {item!r}'
                )

            # We don't support steps
            if i.step is not None:
                raise IndexError(f'Step is not supported.')

            # All slices except for the last one must not have a start and stop
            # value set. The leading dimensions of all arrays must be equal.
            if idx < length - 1 and (i.start is not None or i.stop is not None):
                raise IndexError(
                    f'Cannot partially set values in the leading dimensions '
                    f'({item[:-1]!r} at dimension {idx})'
                )

        # We only need to handle the last element in item, all others are
        # slice(None)
        start, stop = _parse_item(item[-1], self.shape)

        if stop is None or stop - start < 0:
            raise IndexError(f'Could not determine size of slice ({item})')
        length = stop - start

        if np.isscalar(value):
            # Infer the dtype from value if the SparseArray doesn't have a
            # dtype yet
            if self._leading_shape is None and len(item) > 1:
                raise ValueError(
                    'Scalars can only be set if the shape is '
                    'known or the array is one-dimensional'
                )
            value = self._new_full(
                shape=self._leading_shape + (length,), fill_value=value,
                dtype=self.dtype if self.dtype else _dtype_from_value(value)
            )

        if isinstance(value, array_types):
            # The last dimension has to match the length defined by the indexing
            if value.shape[-1] != length:
                raise ValueError(
                    f'Could not assign array with shape {value.shape} to '
                    f'SparseArray slice with size {length}'
                )

            # All other dimensions have to match the current shape
            if value.shape[:-1] != self._leading_shape:
                raise ValueError(
                    f'Shape mismatch: SparseArray has shape {self.shape}, but '
                    f'assigned array has shape {value.shape}.'
                )

            # start cannot be negative here, so we don't need _SparseSegment.from_array
            self._add_segment(_SparseSegment(start, value))
        else:
            raise NotImplementedError()

    def __getitem__(self, item):
        """
        >>> a = zeros(20, dtype=np.float64)
        >>> a[:10] = 1
        >>> a[15:] = np.arange(5, dtype=np.float64) + 1

        # Integer getitem
        >>> a[0], a[9], a[10], a[14], a[15], a[18], a[-1]
        (1.0, 1.0, 0.0, 0.0, 1.0, 4.0, 5.0)
        >>> a[21]
        Traceback (most recent call last):
          ...
        IndexError: Index 21 is out of bounds for ArrayInterval with shape (20,)

        # Slicing
        >>> np.asarray(a[:5])
        array([1., 1., 1., 1., 1.])
        >>> np.asarray(a[10:15])
        array([0., 0., 0., 0., 0.])
        >>> a[-10:]
        SparseArray(_SparseSegment(onset=5, array=array([1., 2., 3., 4., 5.])), shape=(10,))
        >>> np.asarray(a[-10:])
        array([0., 0., 0., 0., 0., 1., 2., 3., 4., 5.])
        >>> np.asarray(a[:-10])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        # Multi-dimensional
        >>> a = SparseArray((2, 10))
        >>> a[..., :5] = np.arange(10).reshape((2, 5))
        >>> a[..., 7:] = 1
        >>> np.asarray(a)
        array([[0, 1, 2, 3, 4, 0, 0, 1, 1, 1],
               [5, 6, 7, 8, 9, 0, 0, 1, 1, 1]])
        >>> a[..., :5]
        SparseArray(_SparseSegment(onset=0, array=array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])), shape=(2, 5))
        >>> a[0]
        SparseArray(_SparseSegment(onset=0, array=array([0, 1, 2, 3, 4])), _SparseSegment(onset=7, array=array([1, 1, 1])), shape=(10,))
        >>> a[-1]
        SparseArray(_SparseSegment(onset=0, array=array([5, 6, 7, 8, 9])), _SparseSegment(onset=7, array=array([1, 1, 1])), shape=(10,))
        >>> a[5]
        Traceback (most recent call last):
         ...
        IndexError: index 5 is out of bounds for axis 0 with size 2
        >>> np.asarray(a[1, 3:5])
        array([8, 9])
        >>> np.asarray(a[..., 0])
        array([0, 5])
        >>> a[..., 6]
        array([0, 0])
        >>> zeros((10, 20))[0, :]
        SparseArray(shape=(20,))
        >>> zeros((10, 20))[1, 2, 3]
        Traceback (most recent call last):
          ...
        IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed

        # Unknown time length
        >>> a = zeros((None,))
        >>> a[10]
        0.0
        >>> a[:10], np.asarray(a[:10])
        (SparseArray(shape=(10,)), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32))
        >>> a[:5] = 1
        >>> np.asarray(a[2:7])
        array([1., 1., 1., 0., 0.], dtype=float32)
        >>> a[0]
        1.0
        >>> a[10]
        0.0
        >>> a.persist_shape()
        (5,)
        >>> a[10]
        Traceback (most recent call last):
          ...
        IndexError: Index 10 is out of bounds for ArrayInterval with shape (5,)
        >>> a = zeros((2, None))
        >>> a[:, :5] = np.ones((2, 5), dtype=np.float32)
        >>> a[0]
        SparseArray(_SparseSegment(onset=0, array=array([1., 1., 1., 1., 1.], dtype=float32)), shape=(None,))
        >>> a[:1]
        SparseArray(_SparseSegment(onset=0, array=array([[1., 1., 1., 1., 1.]], dtype=float32)), shape=(1, None))
        >>> a[0, :3]
        SparseArray(_SparseSegment(onset=0, array=array([1., 1., 1.], dtype=float32)), shape=(3,))
        >>> a[0, 3:]
        SparseArray(_SparseSegment(onset=0, array=array([1., 1.], dtype=float32)), shape=(None,))
        """
        item = tuple(_normalize_item(item, self.ndim))

        assert len(item) == self.ndim, (item, self.shape)

        # Check if the last dimension (the sparse dimension) is indexed.
        # If not, use a shortcut
        if item[-1] == slice(None):
            # The sparse dimension is not indexed, so we can simply forward to
            # numpy/torch.
            arr = dataclasses.replace(
                self,
                shape=_shape_for_item(self.shape, item),
                _segments=[dataclasses.replace(s, array=s.array[item]) for s in self._segments],
            )
            return arr

        # Construct a selector for the leading dimensions. Remove the last slice
        # from the item as it is handled differently
        *selector, item = item
        selector = tuple(selector)

        # If the last dimension is indexed with an integer, find the correct
        # segment and return a numpy array
        if isinstance(item, (int, np.integer)):
            index = item
            if index < 0:
                index = index + self._time_length
            if index < 0 or self._time_length is not None and index > self._time_length:
                raise IndexError(
                    f'Index {item} is out of bounds for ArrayInterval with '
                    f'shape {self.shape}'
                )

            # Get the segment that could overlap with the new segment. The
            # segments are sorted, so bisect finds the only candidate.
            position = bisect.bisect_right([s.onset for s in self._segments], index)

            if position > 0:
                s = self._segments[position - 1]
                if index < s.offset:
                    return s.array[selector + (index - s.onset,)]

            # Return the fill value if not segment overlaps with the index, r
            return self._new_full(
                shape=_shape_for_item(self._leading_shape, selector)
            )

        # Get clipped start/stop values from the function used in ArrayInterval
        start, stop = _parse_item(item, self.shape)

        # This is numpy behavior: return an empty array if stop <= start
        if stop is not None and stop <= start:
            return np.zeros(0, dtype=self.dtype)

        # Find all segments that overlap with the requested slice
        active = [
            s for s in self._segments if (stop is None or s.onset < stop) and s.offset > start
        ]
        if stop is None:
            time_length = None
        else:
            time_length = int(stop - start)
        shifted_segments = []
        for s in active:
            # Slicing of the last dimension is done in _cut
            shifted_segments.append(_SparseSegment.from_array(
                s.array[selector + (slice(None),)],
                s.onset - start,
                time_length
            ))
        arr = dataclasses.replace(
            self,
            shape=_shape_for_item(self._leading_shape, selector) + (time_length,),
            _segments=shifted_segments,
        )
        return arr

    def __len__(self):
        return self.shape[0]

    # Useful math operations
    def __add__(self, other):
        """
        >>> a = SparseArray(10)
        >>> a[:5] = 1
        >>> b = SparseArray(10)
        >>> b[7:] = 2
        >>> np.asarray(a + b)
        array([1, 1, 1, 1, 1, 0, 0, 2, 2, 2])
        >>> c = np.arange(10)
        >>> a + c
        array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
        >>> c + a
        array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])

        >>> a + SparseArray(5)
        Traceback (most recent call last):
         ...
        ValueError: Shape mismatch: (10,) (5,)
        >>> c + SparseArray(5)
        Traceback (most recent call last):
         ...
        ValueError: Shape mismatch: (10,) (5,)
        """
        if isinstance(other, SparseArray):
            _check_shape(self.shape, other.shape)
            self_copy = copy.copy(self)
            self_copy += other
            return self_copy
        else:
            # Let numpy handle this case
            return NotImplemented

    def __iadd__(self, other):
        """
        >>> a = SparseArray(10)
        >>> a[:5] = 1
        >>> b = SparseArray(10)
        >>> b[7:] = 2
        >>> c = np.arange(10)
        >>> a += b
        >>> a
        SparseArray(_SparseSegment(onset=0, array=array([1, 1, 1, 1, 1])), _SparseSegment(onset=7, array=array([2, 2, 2])), shape=(10,))
        >>> c += b
        >>> c
        array([ 0,  1,  2,  3,  4,  5,  6,  9, 10, 11])
        >>> a += c
        Traceback (most recent call last):
         ...
        TypeError: <class 'numpy.ndarray'>
        """
        if isinstance(other, SparseArray):
            _check_shape(self.shape, other.shape)
            for s in other._segments:
                self._add_segment(copy.copy(s))
        else:
            raise TypeError(type(other))
        return self

    def __mul__(self, other):
        """
        >>> a = SparseArray(10)
        >>> a[:5] = 1
        >>> a * 2
        SparseArray(_SparseSegment(onset=0, array=array([2, 2, 2, 2, 2])), shape=(10,))
        >>> a._pad_value = np.full((1,), 2, dtype=a.dtype)
        >>> np.asarray(a*2)
        array([2, 2, 2, 2, 2, 4, 4, 4, 4, 4])
        >>> 2 * a
        SparseArray(_SparseSegment(onset=0, array=array([2, 2, 2, 2, 2])), pad_value=4, shape=(10,))
        >>> np.arange(10) * a
        array([ 0,  1,  2,  3,  4, 10, 12, 14, 16, 18])
        """
        if isinstance(other, (float, int, complex)):
            # Scalar value, multiply everything by it
            arr = dataclasses.replace(
                self, _segments=[
                    dataclasses.replace(s, array=s.array * other)
                    for s in self._segments
                ],
                _pad_value=self._pad_value * other
            )
            return arr
        else:
            # Let numpy find a way to handle this
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (float, int, complex)):
            # We can safely change the order for scalar multiplication
            return self * other
        else:
            return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Support cheap elementwise array operations with numpy arrays by only
        evaluating the function where data is present.
        Always returns a `np.ndarray`, never a `SparseArray`.

        >>> a = SparseArray(10)
        >>> a[:8] = np.arange(8)
        >>> c = -np.arange(10)
        >>> c
        array([ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9])

        # >>> a * c
        array([  0,  -1,  -4,  -9, -16, -25, -36, -49,   0,   0])
        >>> c * a
        array([  0,  -1,  -4,  -9, -16, -25, -36, -49,   0,   0])
        >>> a * c
        array([  0,  -1,  -4,  -9, -16, -25, -36, -49,   0,   0])
        >>> c
        array([ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9])
        >>> c -= a
        >>> c
        array([  0,  -2,  -4,  -6,  -8, -10, -12, -14,  -8,  -9])
        >>> c + a
        array([ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9])
        >>> c += a
        >>> d = np.ones((1, 10))
        >>> np.add(c, a, out=d)
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -8., -9.]])
        >>> d
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -8., -9.]])
        """
        # Only supported if all segments are numpy arrays
        if self.is_torch:
            return NotImplemented

        # Only support elementwise operations
        # TODO: We could allow reductions along free dimensions
        if method != '__call__':
            return NotImplemented
        if len(inputs) != 2:
            return NotImplemented

        # Only support a small subset of elementwise operations for now.
        # Other operators can be tricky to implement and may result in
        # unexpected behavior
        if ufunc not in [np.add, np.subtract, np.multiply, np.divide]:
            return NotImplemented

        # Only support combinations of np.ndarray and SparseArray
        if isinstance(inputs[0], SparseArray) and isinstance(inputs[1], SparseArray):
            return NotImplemented

        # The combination function used below only supports np.ndarray on LHS
        # and SparseArray on RHS. The supported operations allow changing the
        # order or the operands to support SparseArray on LHS
        if isinstance(inputs[1], np.ndarray) and isinstance(inputs[0], SparseArray):
            # Inplace SparseArray + np.ndarray would change the variable type,
            # we don't want that
            if out is not None and out[0] is inputs[0]:
                raise TypeError()

            # Change the order of the operands
            inputs = inputs[::-1]
            _ufunc = ufunc
            ufunc = lambda a, b, **kwargs: _ufunc(b, a, **kwargs)

        # Make sure that both have the same shape
        _check_shape(inputs[0].shape, inputs[1].shape)

        if out is not None:
            # Out is a tuple in numpy, but we only support operations with one
            # output array
            assert len(out) == 1, out
            out = out[0]

        # Left-hand side is numpy array, right hand side is SparseArray,
        return _combine_inplace_array_with_sparse(ufunc, inputs[0], inputs[1], out=out)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Support cheap elementwise array operations for torch tensors.

        Note:
            Clones the input tensor and uses inplace operations.
            IT DOES NOT SUPPORT GRADIENTS RELIABLY!
            TODO: Test this

        >>> a = SparseArray(10)
        >>> a[:8] = torch.arange(8)
        >>> c = -torch.arange(10)
        >>> c -= a
        >>> c
        tensor([  0,  -2,  -4,  -6,  -8, -10, -12, -14,  -8,  -9])
        >>> c + a
        tensor([ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9])

        Some functions don't work in some torch versions
        # >>> c * a
        # >>> a / c
        # >>> a * c
        tensor([  0,  -1,  -4,  -9, -16, -25, -36, -49,   0,   0])
        """
        import torch
        if kwargs is not None:
            return NotImplemented

        # Only supported if tensor is RHS
        if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], SparseArray):
            return NotImplemented

        # Only supported if all segments are torch tensors
        if not args[1].is_torch:
            return NotImplemented

        # Only support a small subset of elementwise operations for now.
        # Other operators can be tricky to implement and may result in
        # unexpected behavior
        if func not in (
                torch.Tensor.add, torch.Tensor.add_,
                torch.Tensor.sub, torch.Tensor.sub_,
        ):
            return NotImplemented

        # If func is not inplace: Copy. All inplace function names end with '_'
        if not func.__name__.endswith('_'):
            out = None  # Clones _combine_inplace_array_with_sparse
            func_ = getattr(torch.Tensor, func.__name__ + '_')
        else:
            out = args[0]  # No clone
            func_ = func

        def func(a, b, out):
            # The contents of a and out are always equal due to the way the
            # inputs are arranged, but out can be a copy, so always use out
            # as first arg
            func_(out, b)

        return _combine_inplace_array_with_sparse(
            func, args[0], args[1], out=out
        )


def _combine_sparse_arrays(func, input1, input2):
    """Function to compute any elementwise operation over two Sparse Arrays."""
    assert input1.shape == input2.shape
    connected_components = []
    segments = []
    for segment in input1._segments:
        segments.append(segment)
    for segment in input2._segments:
        segments.append(segment)
    segments = sorted(segments, key=lambda x: x.onset)
    c = []
    for s in segments:
        if len(c) == 0:
            c = [s.onset, s.offset]
            continue
        if c[-1] >= s.onset:
            c[-1] = max(s.offset, c[-1])
        else:
            connected_components.append(c)
            c = [s.onset, s.offset]
    if c:
        connected_components.append(c)

    out = input1.__class__(
        shape=input1.shape,
        _pad_value=func(input1._pad_value, input2._pad_value)
    )
    for start, stop in connected_components:
        lhs = np.asarray(input1[..., start:stop])
        rhs = input2[..., start:stop]
        result = func(lhs, rhs)
        out.add_segment(start, result)
    return out


def _combine_inplace_array_with_sparse(func, array, sparse_array, out=None):
    if out is None:
        out = copy.deepcopy(array)
    elif array is not out:
        out[...] = array

    for segment in sparse_array._segments:
        func(array[..., segment.onset:segment.offset], segment.array,
             out=out[..., segment.onset:segment.offset])

    # Shortcut functions where 0 is the neutral element
    if sparse_array.pad_value == 0 and \
            func in shortcut_operations:
        return out

    # Handle everything outside segments
    intervals = cy_invert_intervals(sparse_array.interval.normalized_intervals, sparse_array.shape[-1])
    for start, stop in intervals:
        func(array[..., start:stop], sparse_array.pad_value,
             out=out[..., start:stop])
    return out
