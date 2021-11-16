import bisect
import copy
from dataclasses import dataclass, field

from typing import Tuple, Any

from cached_property import cached_property

from paderbox.array.interval.core import ArrayInterval
import numpy as np
import paderbox as pb
from paderbox.array.interval.util import cy_parse_item, cy_invert_intervals
import torch


def _normalize_shape(shape):
    if shape is None:
        # As of now, we only support fixed shapes. This can be changed
        raise TypeError(f'None shape is not supported')

    if isinstance(shape, int):
        return shape,

    if not isinstance(shape, (tuple, list)):
        raise TypeError(f'Invalid shape {shape} of type {type(shape)}')

    if not all(isinstance(s, int) for s in shape):
        raise TypeError(
            f'Invalid shape {shape} with elements of type {type(shape[0])}'
        )

    return tuple(shape)


def _shape_for_index(shape, index):
    if shape == ():
        return ()
    return np.broadcast_to(np.ones(1), shape)[index].shape


def zeros(shape):
    return SparseArray(shape=shape)


def _check_shape(shape1, shape2):
    if shape1 != shape2:
        raise ValueError(f'Shape mismatch: {shape1} {shape2}')


def _cut(array, onset, target_length):
    offset = onset + array.shape[-1]
    if onset < 0:
        array = array[..., -onset:]
        onset = 0
    if offset > target_length:
        array = array[..., :target_length - onset]
    return onset, array


# Use dataclass for recursive to_numpy
@dataclass
class _SparseSegment:
    onset: int
    array: [np.ndarray, torch.Tensor]

    @property
    def offset(self):
        return self.onset + self.array.shape[-1]

    @property
    def segment_length(self):
        return self.array.shape[-1]

    @cached_property
    def is_torch(self):
        return isinstance(self.array, torch.Tensor)


@dataclass
class SparseArray:
    """
    A sparse array implementation using intervals to define non-zero regions.

    >>> a = SparseArray(shape=(20, ))
    >>> a
    SparseArray(shape=(20,))
    >>> a[5:10] = np.ones(5, dtype=int)
    >>> a
    SparseArray(_SparseSegment(onset=5, array=array([1, 1, 1, 1, 1])), shape=(20,))
    >>> np.asarray(a)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> a[15:20] = 2
    >>> a
    SparseArray(_SparseSegment(onset=5, array=array([1, 1, 1, 1, 1])), _SparseSegment(onset=15, array=array([2, 2, 2, 2, 2])), shape=(20,))
    >>> np.asarray(a)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2])

    Overlapping intervals are not allowed
    >>> a[14:16] = 3
    Traceback (most recent call last):
      ...
    ValueError: Overlap detected between
      _SparseSegment(onset=15, array=array([2, 2, 2, 2, 2]))
    and newly added
      _SparseSegment(onset=14, array=array([3, 3]))


    dtype is inferred from segments:
    >>> a.dtype
    dtype('int64')
    >>> a.pad_value, type(a.pad_value)
    (0, <class 'numpy.int64'>)

    Can't add segments with differing dtypes
    >>> a[:3] = np.ones(3, dtype=np.float32)
    Traceback (most recent call last):
      ...
    TypeError: Type mismatch: SparseArray has dtype int64, but assigned array has dtype float32

    Adding multiple SparseArray returns a SparseArray
    >>> b = SparseArray(a.shape)
    >>> b[10:15] = 8
    >>> a + b
    SparseArray(_SparseSegment(onset=5, array=array([1, 1, 1, 1, 1])), _SparseSegment(onset=10, array=array([8, 8, 8, 8, 8])), _SparseSegment(onset=15, array=array([2, 2, 2, 2, 2])), shape=(20,))
    >>> np.asarray(a + b)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 2, 2, 2, 2, 2])

    Arithmetic works with numpy arrays
    >>> np.ones(20, dtype=int) * a
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    >>> np.ones(20, dtype=int) + a
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
    >>> a + np.ones(20, dtype=int)
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
    >>> c = np.ones(20, dtype=int)
    >>> c += a
    >>> c
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3])

    """
    # Public constructor interface
    shape: tuple

    # Private constructor arguments: Don't use _segments directly.
    # This is required for pt.data.batch.example_to_device to work
    _segments: list = field(default_factory=list)

    # Other instance attributes
    pad_value = 0.
    _is_torch = False
    dtype = None

    def __post_init__(self):
        self.shape = _normalize_shape(self.shape)
        self._update_dtype(self.dtype)

    @classmethod
    def from_array_and_onset(cls, array, onset, shape):
        out = SparseArray(shape=shape)
        if onset < 0:
            onset = 0
            array = array[..., -onset:]
        if array.shape[-1] > out.shape[-1]:
            array = array[..., out.shape[-1] - onset]
        out._add_segment(_SparseSegment(onset, array))
        return out

    def as_contiguous(self, dtype=None):
        arr = self._new_full(dtype=dtype)
        for segment in self._segments:
            arr[..., segment.onset:segment.offset] = segment.array
        return arr

    def _update_dtype(self, dtype, pad_value=0.):
        self.dtype = dtype
        if dtype is None and self._segments:
            self.dtype = self._segments[0].array.dtype
        if self.dtype is not None:
            self._is_torch = isinstance(self.dtype, torch.dtype)
            if self._is_torch:
                self.pad_value = torch.tensor(pad_value, dtype=self.dtype)
            else:
                self.pad_value = self.dtype.type(pad_value)
        if self._segments:
            assert all([s.array.dtype == self.dtype for s in self._segments]), (
                [s.array.dtype for s in self._segments], self.dtype)

    @property
    def interval(self) -> ArrayInterval:
        ai = pb.array.interval.zeros(self.shape[-1])
        s: _SparseSegment
        for s in self._segments:
            ai[s.onset:s.offset] = True
        return ai

    def _new_full(self, *, shape=None, dtype=None, fill_value=None):
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if fill_value is None:
            fill_value = self.pad_value
        if self._is_torch:
            return torch.full(
                size=shape, fill_value=fill_value, dtype=dtype
            )
        else:
            return np.full(
                shape=shape, fill_value=fill_value, dtype=dtype
            )

    def _add_segment(self, segment: _SparseSegment):
        if self._segments and (
                self._is_torch and not isinstance(segment.array, torch.Tensor)
                or not isinstance(segment.array, np.ndarray)
        ):
            raise TypeError(type(segment.array))

        # Check input dtype and shape
        if self.dtype is None:
            # This is the first segment we add. It defines dtype and pad value
            self._update_dtype(segment.array.dtype)
        elif self.dtype != segment.array.dtype:
            raise TypeError(
                f'Type mismatch: SparseArray has dtype {self.dtype}, but '
                f'assigned array has dtype {segment.array.dtype}'
            )

        # The leading dimensions must be equal
        if segment.array.shape[:-1] != self.shape[:-1]:
            raise ValueError(
                f'Shape mismatch: SparseArray has shape {self.shape}, but '
                f'assigned array has shape {segment.array.shape}.'
            )

        if segment.offset < 0 or segment.onset >= self.shape[-1]:
            # Ignore setting anything that is outside of the boundaries (numpy
            # behavior)
            return

        # Get insert position
        position = bisect.bisect_right(
            [s.onset for s in self._segments],
            segment.onset
        )

        # Check for overlaps with neighboring segments
        if (
                len(self._segments) > 0
                and position > 0
                and self._segments[position - 1].offset > segment.onset
        ):
            import textwrap
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
            import textwrap
            raise ValueError(
                f'Overlap detected between\n'
                f'{textwrap.indent(repr(self._segments[position]), "  ")}\n'
                f'and newly added\n'
                f'{textwrap.indent(repr(segment), "  ")}'
            )

        self._segments.insert(position, segment)

    def __copy__(self):
        return self.__class__(
            shape=self.shape, _segments=[
                _SparseSegment(segment.onset, segment.array)
                for segment in self._segments
            ]
        )

    def __array__(self, dtype=None):
        if self._is_torch:
            raise NotImplementedError(
                '__array__ is not implemented for torch SparseArrays'
            )
        return self.as_contiguous(dtype)

    def __repr__(self):
        if self._segments:
            content = ', '.join(map(str, self._segments))
            content = content + ', '
        else:
            content = ''
        if self.pad_value != 0:
            p = f'pad_value={self.pad_value}, '
        else:
            p = ''
        return f'{self.__class__.__name__}({content}{p}shape={self.shape})'

    def _repr_pretty_(self, p, cycle):
        """
        >>> pb.utils.pretty.pprint(SparseArray(10))
        SparseArray(shape=(10,))
        >>> a = SparseArray(10)
        >>> a[:5] = 1
        >>> a[7:] = 2
        >>> pb.utils.pretty.pprint(a)
        SparseArray(_SparseSegment(onset=0, array=array([1, 1, 1, 1, 1])),
                    _SparseSegment(onset=7, array=array([2, 2, 2])),
                    shape=(10,))
        >>> a.pad_value=-1
        >>> pb.utils.pretty.pprint(a)
        SparseArray(_SparseSegment(onset=0, array=array([1, 1, 1, 1, 1])),
                    _SparseSegment(onset=7, array=array([2, 2, 2])),
                    shape=(10,), pad_value=-1)
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
        NotImplementedError: Only ellipsis is allowed for first dimensions in setitem


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
        """
        if isinstance(item, tuple):
            selector = item[:len(self.shape) - 1]
            # Complex indexing is not supported in __setitem__
            if len(selector) != 1 or selector[0] is not ...:
                raise NotImplementedError(
                    'Only ellipsis is allowed for first dimensions in setitem'
                )

            item = item[-1]
        if not isinstance(item, slice):
            raise NotImplementedError(
                f'{self.__class__.__name__}.__setitem__ only supports slices '
                f'for indexing, not {item!r}'
            )
        start, stop = cy_parse_item(item, self.shape)
        length = stop - start

        if np.isscalar(value):
            value = self._new_full(shape=self.shape[:-1] + (length,), fill_value=value)

        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if value.shape[-1] != length:
                raise ValueError(
                    f'Could not assign array with shape {value.shape} to '
                    f'SparseArray slice with size {length}'
                )

            self._add_segment(_SparseSegment(start, value))
        else:
            raise NotImplementedError()

    def __getitem__(self, item):
        """
        >>> a = SparseArray(20)
        >>> a[:10] = 1
        >>> a[15:] = np.arange(5) + 1

        # Integer getitem
        >>> a[0], a[9], a[10], a[14], a[15], a[18], a[-1]
        (1, 1, 0, 0, 1, 4, 5)
        >>> a[21]
        Traceback (most recent call last):
          ...
        IndexError: Index 21 is out of bounds for ArrayInterval with shape (20,)

        # Slicing
        >>> np.asarray(a[:5])
        array([1, 1, 1, 1, 1])
        >>> np.asarray(a[10:15])
        array([0, 0, 0, 0, 0])
        >>> a[-10:]
        SparseArray(_SparseSegment(onset=5, array=array([1, 2, 3, 4, 5])), shape=(10,))
        >>> np.asarray(a[-10:])
        array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        >>> np.asarray(a[:-10])
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

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
        """
        if not isinstance(item, tuple):
            item = (item,)

        assert len(item) <= len(self.shape), (item, self.shape)

        # Check if the last dimension (the sparse dimension) is indexed.
        # If not, use a shortcut
        if (
                # Ellipsis is present and the last element
                (... in item and item[-1] is ...)
                or
                # Ellipsis is not present and no index is present for the last
                # dimension or the index is slice(None) (select everything)
                (... not in item and (len(item) < len(self.shape) or item[-1] == slice(None)))
        ):
            # The sparse dimension is not indexed, so we can simply forward to
            # numpy/torch
            arr = self.__class__(
                shape=self._segments[0].array[item].shape[:-1] + (self.shape[-1],),
                _segments=[_SparseSegment(s.onset, s.array[item]) for s in self._segments]
            )
            return arr

        # Construct a selector for the leading dimensions. Remove the last slice
        # from the item as it is handled differently
        *selector, item = item
        selector = tuple(selector)

        # If the last dimension is indexed with an integer, find the correct
        if isinstance(item, (int, np.integer)):
            index = item
            if index < 0:
                index = index + self.shape[-1]
            if index < 0 or self.shape is not None and index > self.shape[-1]:
                raise IndexError(
                    f'Index {item} is out of bounds for ArrayInterval with '
                    f'shape {self.shape}'
                )
            # Could be optimized
            position = bisect.bisect_right([s.onset for s in self._segments], index)

            # position == 0 is before the first segment, so padded
            if position == 0:
                if len(self.shape) == 1:
                    # Return scalar
                    return self.pad_value
                else:
                    # Return array
                    return self._new_full(
                        shape=_shape_for_index(self.shape[:-1], selector)
                    )
            s = self._segments[position - 1]
            # onset <= index is always true here
            if index < s.offset:
                return s.array[selector + (index - s.onset,)]
            else:
                if len(self.shape) == 1:
                    # Return scalar
                    return self.pad_value
                else:
                    # Return array
                    return self._new_full(
                        shape=_shape_for_index(self.shape[:-1], selector)
                    )

        # Get clipped start/stop values from the function used in ArrayInterval
        start, stop = cy_parse_item(item, self.shape)

        # This is numpy behavior: return an empty array if stop <= start
        if stop <= start:
            return np.zeros(0, dtype=self.dtype)

        # Find all segments that overlap with the requested slice
        active = [
            s for s in self._segments if s.onset < stop and s.offset > start
        ]
        time_length = int(stop - start)
        shifted_segments = []
        for s in active:
            # Slicing of the last dimension is done in _cut
            onset, array = _cut(s.array[selector + (slice(None),)], s.onset - start, time_length)
            shifted_segments.append(
                _SparseSegment(onset, array)
            )
        arr = self.__class__(
            shape=(_shape_for_index(self.shape[:-1], selector) + (time_length,)),
            _segments=shifted_segments
        )
        arr._update_dtype(self.dtype, self.pad_value)
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
                self._add_segment(_SparseSegment(s.onset, s.array))
        else:
            raise TypeError(type(other))
        return self

    def __mul__(self, other):
        """
        >>> a = SparseArray(10)
        >>> a[:5] = 1
        >>> a * 2
        SparseArray(_SparseSegment(onset=0, array=array([2, 2, 2, 2, 2])), shape=(10,))
        >>> a.pad_value = 2
        >>> np.asarray(a*2)
        array([2, 2, 2, 2, 2, 4, 4, 4, 4, 4])
        >>> 2 * a
        SparseArray(_SparseSegment(onset=0, array=array([2, 2, 2, 2, 2])), pad_value=4, shape=(10,))
        >>> np.arange(10) * a
        array([ 0,  1,  2,  3,  4, 10, 12, 14, 16, 18])
        """
        if isinstance(other, (np.float, np.int, np.complex)):
            # Scalar value, multiply everything by it
            arr = SparseArray(
                shape=self.shape,
                _segments=[
                    _SparseSegment(s.onset, s.array * other)
                           for s in self._segments
                ]
            )
            arr._update_dtype(self.dtype, self.pad_value * other)
            return arr
        else:
            # Let numpy find a way to handle this
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (np.float, np.int, np.complex)):
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
        if self._is_torch:
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
            raise NotImplemented

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
        if kwargs is not None:
            return NotImplemented

        # Only supported if tensor is RHS
        if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], SparseArray):
            return NotImplemented

        # Only supported if all segments are torch tensors
        if not args[1]._is_torch:
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
            out = args[0]   # No clone
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

    out = input1.__class__(shape=input1.shape)
    for start, stop in connected_components:
        lhs = np.asarray(input1[..., start:stop])
        rhs = input2[..., start:stop]
        result = func(lhs, rhs)
        out.add_segment(start, result)
    out._update_dtype(func(input1.pad_value, input2.pad_value))
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
            func in [np.add, np.subtract, torch.Tensor.add_, torch.Tensor.add,
                torch.Tensor.sub_, torch.Tensor.sub]:
        return out

    # Handle everything outside segments
    intervals = cy_invert_intervals(sparse_array.interval.normalized_intervals, sparse_array.shape[-1])
    for start, stop in intervals:
        func(array[..., start:stop], sparse_array.pad_value,
             out=out[..., start:stop])
    return out
