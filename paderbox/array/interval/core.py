"""
`ArrayInterval` offers a user-readable object for stashing activity information.
In combination with jsonpickle this allows for a low resource possibility to
save activity information for large time streams.
"""

import operator
from typing import Optional, Union, Iterable

import numpy as np
from paderbox.array.interval.util import (
    cy_non_intersection,
    cy_intersection,
    cy_parse_item,
    cy_str_to_intervals,
    cy_invert_intervals,
)


def ArrayInterval_from_str(string, shape, inverse_mode=False) -> 'ArrayInterval':
    """
    >>> ArrayInterval_from_str('1:4, 5:20, 21:25', shape=50)
    ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
    >>> ArrayInterval_from_str('1:4', shape=50)
    ArrayInterval("1:4", shape=(50,))
    >>> ArrayInterval_from_str('1:4,', shape=50)
    ArrayInterval("1:4", shape=(50,))
    >>> ArrayInterval_from_str('0:142464640,', shape=242464640)
    ArrayInterval("0:142464640", shape=(242464640,))

    """
    ai = zeros(shape)
    if string == '':
        pass
    else:
        if not ',' in string:
            string = string + ','
        try:
            ai.add_intervals_from_str(string)
        except Exception as e:
            raise Exception(string) from e
    ai.inverse_mode = inverse_mode
    return ai


def ArrayInterval_from_pairs(pairs: 'list[list[int]]', shape=None, inverse_mode=False):
    """
    >>> ArrayInterval_from_pairs([[1, 4], [5, 20], [21, 25]], shape=50)
    ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
    >>> ArrayInterval_from_pairs([[1, 4]], shape=50)
    ArrayInterval("1:4", shape=(50,))
    >>> ArrayInterval_from_pairs([[1, 4]], shape=50)
    ArrayInterval("1:4", shape=(50,))
    >>> ArrayInterval_from_pairs([[0, 142464640]], shape=242464640)
    ArrayInterval("0:142464640", shape=(242464640,))

    """
    ai = zeros(shape)
    ai.add_intervals([slice(start, end) for start, end in pairs])
    ai.inverse_mode = inverse_mode
    return ai


def intervals_to_str(intervals):
    return ', '.join(f'{start}:{end}' for start, end in intervals)


def _normalize_shape(shape):
    if shape is None:
        return None

    if isinstance(shape, int):
        return shape,

    if not isinstance(shape, (tuple, list)):
        raise TypeError(f'Invalid shape {shape} of type {type(shape)}')

    # As of now, we only support 1D. We haven't decided yet how to handle
    # higher numbers of dimensions as it is unclear what they mean. Probably
    # the last dimension will remain as it is and the earlier dimensions will
    # be independent dimensions.
    if not len(shape) == 1:
        raise ValueError(f'Invalid shape {shape} has to have length 1')
    if not isinstance(shape[-1], int):
        raise TypeError(
            f'Invalid shape {shape} with elements of type {type(shape[0])}'
        )

    return tuple(shape)


def zeros(shape: Optional[Union[int, tuple, list]] = None) -> 'ArrayInterval':
    """
    Instantiate an `ArrayInterval` filled with zeros.

    Note: The difference from numpy is that the argument shape is optional.
          When shape is None, some operations aren't supported because the
          length is unknown.
          e.g. array_interval[:] fails because the length is unknown, while
               array_interval[:1000] works.

    Args:
        shape: `None`, `int` or `tuple`/`list` that contains one `int`.

    Returns:
        `ArrayInterval` filled with zeros

    Examples:
        >>> ai = zeros(10)
        >>> ai
        ArrayInterval("", shape=(10,))
        >>> ai[2:3] = 1
        >>> ai
        ArrayInterval("2:3", shape=(10,))
        >>> ai[:]  # getitem converts the ArrayInterval to numpy
        array([False, False,  True, False, False, False, False, False, False,
               False])

        >>> ai = zeros()
        >>> ai
        ArrayInterval("", shape=None)
        >>> ai[2:3] = 1
        >>> ai
        ArrayInterval("2:3", shape=None)
        >>> ai[:]
        Traceback (most recent call last):
        ...
        RuntimeError: You tried to slice an ArrayInterval with unknown shape without a stop value.
        This is not supported, either the shape has to be known
        or you have to specify a stop value for the slice (i.e. array_interval[:stop])
        You called the array interval with:
            array_interval[slice(None, None, None)]
        >>> ai[:10]  # getitem converts the ArrayInterval to numpy
        array([False, False,  True, False, False, False, False, False, False,
               False])

    """
    ai = ArrayInterval.__new__(ArrayInterval)
    ai.shape = shape
    return ai


def ones(shape: Optional[Union[int, tuple, list]] = None) -> 'ArrayInterval':
    """
    Instantiate an `ArrayInterval` filled with ones.

    Note: The difference from numpy is that the argument shape is optional.
          When `shape` is `None`, some operations aren't supported because the
          length is unknown.
          e.g. array_interval[:] fails because the length is unknown, while
               array_interval[:1000] works.

        Args:
        shape: `None`, `int` or `tuple`/`list` that contains one `int`.

    Returns:
        `ArrayInterval` filled with ones

    Examples:
        >>> ai = ones(10)
        >>> ai
        ArrayInterval("", shape=(10,), inverse_mode=True)
        >>> ai[2:3] = 0
        >>> ai
        ArrayInterval("2:3", shape=(10,), inverse_mode=True)
        >>> ai[:]  # getitem converts the ArrayInterval to numpy
        array([ True,  True, False,  True,  True,  True,  True,  True,  True,
                True])

        >>> ai = ones()
        >>> ai
        ArrayInterval("", shape=None, inverse_mode=True)
        >>> ai[2:3] = 0
        >>> ai
        ArrayInterval("2:3", shape=None, inverse_mode=True)
        >>> ai[:]
        Traceback (most recent call last):
        ...
        RuntimeError: You tried to slice an ArrayInterval with unknown shape without a stop value.
        This is not supported, either the shape has to be known
        or you have to specify a stop value for the slice (i.e. array_interval[:stop])
        You called the array interval with:
            array_interval[slice(None, None, None)]
        >>> ai[:10]  # getitem converts the ArrayInterval to numpy
        array([ True,  True, False,  True,  True,  True,  True,  True,  True,
                True])
    """
    ai = ArrayInterval.__new__(ArrayInterval)
    ai.inverse_mode = True
    ai.shape = shape
    return ai


class ArrayInterval:
    from_str = staticmethod(ArrayInterval_from_str)
    inverse_mode = False

    def __init__(self, array, *, inverse_mode: bool = False):
        """
        The `ArrayInterval` is in many cases equivalent to a 1-dimensional
        boolean numpy array that stores activity information in an efficient
        way.

        Args:
            array: 
            inverse_mode:
                Internal flag to indicate what the intervals represent:
                 - `False`: Intervals represent `True`
                 - `True`: Intervals represent `False`
                This flag is necessary when the shape is unknown.
                The user does not need to care about this flag. The default is
                fine.

        Examples:
            >>> ai = ArrayInterval(np.array([1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=bool))
            >>> ai
            ArrayInterval("0:2, 3:4, 6:8", shape=(9,))
            >>> ai[:]
            array([ True,  True, False,  True, False, False,  True,  True, False])
            >>> a = np.array([1, 1, 1, 1], dtype=bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([0, 0, 0, 0], dtype=bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([0, 1, 1, 0], dtype=bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([1, 0, 0, 1], dtype=bool)
            >>> assert all(a == ArrayInterval(a)[:])

        """
        if isinstance(array, ArrayInterval):
            self._shape = array.shape
            self.inverse_mode = array.inverse_mode
            self.intervals = array.intervals
        else:
            array = np.asarray(array)
            if array.ndim != 1:
                raise ValueError(
                    f'Only 1-dimensional arrays can be converted to '
                    f'ArrayInterval, not {array!r} with ndim={array.ndim}'
                )
            if array.dtype != bool:
                raise ValueError(
                    f'Only boolean array can be converted to ArrayInterval, not'
                    f'{array!r} with dtype={array.dtype}'
                )

            if inverse_mode:
                array = np.logical_not(array)
            diff = np.diff(array.astype(np.int32))

            rising = list(np.atleast_1d(np.squeeze(np.where(diff > 0), axis=0)) + 1)
            falling = list(np.atleast_1d(np.squeeze(np.where(diff < 0), axis=0)) + 1)

            if array[0] == 1:
                rising = [0] + rising

            if array[-1] == 1:
                falling = falling + [len(array)]

            # ai = ArrayInterval(shape=array.shape)
            self.inverse_mode = inverse_mode
            self._shape = array.shape

            if inverse_mode:
                for start, stop in zip(rising, falling):
                    self[start:stop] = 0
            else:
                for start, stop in zip(rising, falling):
                    self[start:stop] = 1

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = _normalize_shape(shape)
        if self._intervals and self._shape is not None:
            assert self.normalized_intervals[-1][-1] <= self._shape[-1], (shape, self._shape, self.normalized_intervals)

    def __copy__(self):
        if self.inverse_mode:
            ai = ones(shape=self.shape)
        else:
            ai = zeros(shape=self.shape)
        ai.intervals = self.intervals
        return ai

    def __array__(self, dtype=bool):
        """
        Special numpy method. This method is used by numpy to cast foreign
        types to numpy.

        Example for the add operation:
            >>> a = np.array([0, 1, 2])
            >>> ai = ArrayInterval([True, False, True])
            >>> ai + a
            array([1, 1, 3])
            >>> ai = zeros()
            >>> ai + a
            Traceback (most recent call last):
            ...
            RuntimeError: You cannot cast an ArrayInterval to numpy
            when the shape is unknown.
        """
        assert dtype == bool, dtype
        if self.shape is None:
            raise RuntimeError(
                f'You cannot cast an {self.__class__.__name__} to numpy\n'
                f'when the shape is unknown.')
        else:
            return self[:]

    def __reduce__(self):
        """
        >>> from IPython.lib.pretty import pprint
        >>> import pickle
        >>> import jsonpickle, json
        >>> from paderbox.array.interval.core import ArrayInterval
        >>> ai = ArrayInterval.from_str('1:4, 5:20, 21:25', shape=50)
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> pickle.loads(pickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> jsonpickle.loads(jsonpickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> pprint(json.loads(jsonpickle.dumps(ai)))
        {'py/reduce': [{'py/function': 'paderbox.array.interval.core.ArrayInterval_from_str'},
          {'py/tuple': ['1:4, 5:20, 21:25', 50, False]}]}

        >>> ai = ArrayInterval.from_str('1:4, 5:20, 21:25', shape=50)
        >>> ai.inverse_mode = True
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,), inverse_mode=True)
        >>> pickle.loads(pickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,), inverse_mode=True)
        >>> jsonpickle.loads(jsonpickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,), inverse_mode=True)
        >>> pprint(json.loads(jsonpickle.dumps(ai)))
        {'py/reduce': [{'py/function': 'paderbox.array.interval.core.ArrayInterval_from_str'},
          {'py/tuple': ['1:4, 5:20, 21:25', 50, True]}]}

        >>> ai = ArrayInterval.from_str('1:4, 5:20, 21:25', shape=None)
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=None)
        >>> pickle.loads(pickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=None)
        >>> jsonpickle.loads(jsonpickle.dumps(ai))
        ArrayInterval("1:4, 5:20, 21:25", shape=None)
        >>> pprint(json.loads(jsonpickle.dumps(ai)))
        {'py/reduce': [{'py/function': 'paderbox.array.interval.core.ArrayInterval_from_str'},
          {'py/tuple': ['1:4, 5:20, 21:25', None, False]}]}
        """
        return self.from_str, (
            self._intervals_as_str,
            self.shape[-1] if self.shape is not None else self.shape,
            self.inverse_mode,
        )

    _intervals_normalized = True
    # _normalized_intervals = ()
    _intervals = ()

    def __len__(self):
        return self.shape[0]

    @property
    def normalized_intervals(self) -> tuple:
        """
        Normalized intervals. Normalized here means that overlapping intervals
        are merged.

        Note:
            Changes the internal representation to the normalized intervals.
        """
        if not self._intervals_normalized:
            self._intervals = self._normalize(self._intervals)
            self._intervals_normalized = True
        return self._intervals

    @property
    def intervals(self) -> tuple:
        """
        A representation of the intervals as tuples of start and end values.
        """
        return self._intervals

    @intervals.setter
    def intervals(self, item):
        self._intervals_normalized = False
        self._intervals = tuple(item)

    @staticmethod
    def _normalize(intervals):
        """
        >>> ArrayInterval._normalize([])
        ()
        >>> ArrayInterval._normalize([(0, 1)])
        ((0, 1),)
        >>> ArrayInterval._normalize([(0, 1), (2, 3)])
        ((0, 1), (2, 3))
        >>> ArrayInterval._normalize([(0, 1), (20, 30)])
        ((0, 1), (20, 30))
        >>> ArrayInterval._normalize([(0, 1), (1, 3)])
        ((0, 3),)
        >>> ArrayInterval._normalize([(0, 1), (1, 3), (3, 10)])
        ((0, 10),)
        """
        intervals = [(s, e) for s, e in sorted(intervals) if s < e]
        for i in range(len(intervals)):
            try:
                s, e = intervals[i]
            except IndexError:
                break
            try:
                next_s, next_e = intervals[i + 1]
                while next_s <= e:
                    e = max(e, next_e)
                    del intervals[i + 1]
                    next_s, next_e = intervals[i + 1]
            except IndexError:
                pass
            finally:
                intervals[i] = (s, e)
        return tuple(intervals)

    @property
    def _intervals_as_str(self):
        return intervals_to_str(self.normalized_intervals)

    def to_serializable(self):
        """
        Exports intervals and length of `ArrayInterval` to a serializable object.
        Intervals are exported as `self._intervals_to_str` to be human readable. 
        Allows easy export of `ArrayIntervals`, e.g. into .json format. 
        
        >>> from IPython.lib.pretty import pprint
        >>> from paderbox.array.interval.core import ArrayInterval
        >>> ai = ArrayInterval.from_str('1:4, 5:20, 21:25', shape=50)
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> ai.to_serializable()
        ('1:4, 5:20, 21:25', (50,))
        """
        intervals = self.normalized_intervals
        if self.inverse_mode:
            intervals = cy_invert_intervals(intervals, self.shape[-1])
        return intervals_to_str(intervals), self.shape

    @staticmethod
    def from_serializable(obj):
        """
        Reverts `to_serializable`.
        Args:
            obj: Object of length 2 with items (str: intervals, shape)
            
        Returns:
            `ArrayInterval` with specified intervals and shape
            
        Example:
        >>> from IPython.lib.pretty import pprint
        >>> from paderbox.array.interval.core import ArrayInterval
        >>> at = ('1:4, 5:20, 21:25', 50)
        >>> at
        ('1:4, 5:20, 21:25', 50)
        >>> ArrayInterval.from_serializable(at)
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        """
        assert len(obj) == 2, f'Expects object of length 2 with items (intervals, shape), got: {obj}'
        return ArrayInterval_from_str(obj[0], shape=obj[1])

    @staticmethod
    def from_pairs(pairs: 'list[list[int]]', shape=None, inverse_mode=False):
        """
        Construct an ArrayInterval from pairs of start, stop values.

        e.g.:

            ai = ArrayInterval.from_pairs([(1, 2), (10, 15)])

        is the same as:

            ai = zeros()
            ai[1:2] = 1
            ai[10:15] = 1

        Args:
            pairs:
            shape:
            inverse_mode:

        Returns:

        """

        return ArrayInterval_from_pairs(pairs, shape, inverse_mode)

    def __repr__(self):
        if self.inverse_mode:
            return f'{self.__class__.__name__}("{self._intervals_as_str}", shape={self.shape}, inverse_mode={self.inverse_mode})'
        else:
            return f'{self.__class__.__name__}("{self._intervals_as_str}", shape={self.shape})'

    def add_intervals_from_str(self, string_intervals: str):
        """
        Adds intervals from a string representation.

        Args:
            string_intervals: Format "<start>:<end>,<start>:<end>..."
        """
        self.intervals = self.intervals + cy_str_to_intervals(string_intervals)

    def add_intervals(self, intervals: Iterable[slice]):
        """
        Adds intervals from a list of slices.

        Equivalent to, but significantly faster than:
            for item in intervals:
                self[item] = 1

        """
        # Short circuit
        self.intervals = self.intervals + tuple(
            [cy_parse_item(i, self.shape) for i in intervals]
        )

    def __setitem__(self, item, value):
        """
        >>> ai = zeros(50)
        >>> ai[10:15] = 1
        >>> ai
        ArrayInterval("10:15", shape=(50,))
        >>> ai[5:10] = 1
        >>> ai
        ArrayInterval("5:15", shape=(50,))
        >>> ai[1:4] = 1
        >>> ai
        ArrayInterval("1:4, 5:15", shape=(50,))
        >>> ai[15:20] = 1
        >>> ai
        ArrayInterval("1:4, 5:20", shape=(50,))
        >>> ai[21:25] = 1
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> ai[10:15] = 1
        >>> ai
        ArrayInterval("1:4, 5:20, 21:25", shape=(50,))
        >>> ai[0:50] = 1
        >>> ai[0:0] = 1
        >>> ai
        ArrayInterval("0:50", shape=(50,))
        >>> ai[3:6]
        array([ True,  True,  True])
        >>> ai[3:6] = np.array([ True,  False,  True])
        >>> ai
        ArrayInterval("0:4, 5:50", shape=(50,))
        >>> ai[10:13] = np.array([ False,  True,  False])
        >>> ai
        ArrayInterval("0:4, 5:10, 11:12, 13:50", shape=(50,))

        >>> ai = zeros(50)
        >>> ai[:] = 1
        >>> ai[10:40] = 0
        >>> ai
        ArrayInterval("0:10, 40:50", shape=(50,))

        >>> ai = zeros(50)
        >>> ai2 = zeros(10)
        >>> ai2[5:10] = 1
        >>> ai[10:20] = ai2
        >>> ai
        ArrayInterval("15:20", shape=(50,))
        >>> ai2 = zeros(20)
        >>> ai2[0:10] = 1
        >>> ai2[15:20] = 1
        >>> ai[10:30] = ai2
        >>> ai
        ArrayInterval("10:20, 25:30", shape=(50,))

        >>> ai[40:60] = ai2
        Traceback (most recent call last):
          ...
        ValueError: Could not broadcast input with length 20 into shape 10
        >>> ai[-20:] = ai2
        >>> ai
        ArrayInterval("10:20, 25:40, 45:50", shape=(50,))

        >>> ai = zeros(20)
        >>> ai[:10] = ones(10)
        >>> ai
        ArrayInterval("0:10", shape=(20,))
        >>> ai = ones(20)
        >>> ai[:10] = zeros(10)
        >>> ai
        ArrayInterval("0:10", shape=(20,), inverse_mode=True)

        """
        if not isinstance(item, slice):
            raise NotImplementedError(
                f'{self.__class__.__name__}.__setitem__ only supports slices '
                f'for indexing, not {item!r}'
            )

        start, stop = cy_parse_item(item, self.shape)

        if np.isscalar(value):
            if value not in (0, 1):
                # Numpy interprets values as boolean even if they are larger
                # than 1. We don't do that here because using other values than
                # boolean (or 0, 1) often indicates a bug or a wrong assumption
                # by the user.
                raise ValueError(
                    f'{self.__class__.__name__} only supports assigning '
                    f'boolean (or 0 or 1) scalar values, not {value!r}'
                )
            value = bool(value)
            if self.inverse_mode:
                value = not value
            if value:
                self.intervals = self.intervals + ((start, stop),)
            else:
                self.intervals = cy_non_intersection((start, stop), self.intervals)
        elif isinstance(value, (tuple, list, np.ndarray, ArrayInterval)):
            if not isinstance(value, ArrayInterval):
                # Inverse mode has to be the same as self.inverse_mode to have
                # matching intervals. The value is inverted in
                # ArrayInterval.__init__ if inverse_mode=True
                value = ArrayInterval(value, inverse_mode=self.inverse_mode)

            if len(value) != stop - start:
                raise ValueError(
                    f'Could not broadcast input with length {len(value)} into '
                    f'shape {stop - start}'
                )
            value_intervals = value.intervals
            if value.inverse_mode != self.inverse_mode:
                value_intervals = cy_invert_intervals(
                    value_intervals, value.shape[-1]
                )
            intervals = cy_non_intersection((start, stop), self.intervals)
            self.intervals = intervals + tuple([
                (s + start, e + start) for s, e in value_intervals
            ])
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__}.__setitem__ not implemented for '
                f'type {type(value)} of {value!r}'
            )

    def __getitem__(self, item):
        """

        >>> ai = zeros(50)
        >>> ai[19:26]
        array([False, False, False, False, False, False, False])
        >>> ai[10:20] = 1
        >>> ai[25:30] = 1
        >>> ai
        ArrayInterval("10:20, 25:30", shape=(50,))
        >>> ai[19:26]
        array([ True, False, False, False, False, False,  True])
        >>> ai[19]
        True
        >>> ai[5]
        False
        >>> ai[49]
        False
        >>> ai[29]
        True
        >>> ai[-25:-20]
        array([ True,  True,  True,  True,  True])

        Get a similar behavior to numpy when indexing outside of shape:
        >>> ai[-1:1]
        array([], dtype=bool)
        >>> ai[-10:]
        array([False, False, False, False, False, False, False, False, False,
               False])
        >>> ai[45:100]
        array([False, False, False, False, False])


        >>> ai = zeros(3)
        >>> list(ai)
        [False, False, False]
        """
        if isinstance(item, (int, np.integer)):
            index = item
            if index < 0:
                if self.shape is None:
                    raise ValueError(
                        f'Negative indices can only be used on ArrayIntervals '
                        f'with a shape! index={index}'
                    )
                index = index + self.shape[-1]
            if index < 0 or self.shape is not None and index >= self.shape[-1]:
                raise IndexError(
                    f'Index {item} is out of bounds for ArrayInterval with '
                    f'shape {self.shape}'
                )
            # Could be optimized
            for s, e in self.normalized_intervals:
                if e > index:
                    return (index >= s) ^ self.inverse_mode
            return self.inverse_mode

        start, stop = cy_parse_item(item, self.shape)

        # This is numpy behavior
        if stop <= start:
            return np.zeros(0, dtype=bool)

        intervals = cy_intersection((start, stop), self.normalized_intervals)

        if self.inverse_mode:
            arr = np.ones(stop - start, dtype=bool)

            for i_start, i_end in intervals:
                arr[i_start - start:i_end - start] = False
        else:
            arr = np.zeros(stop - start, dtype=bool)

            for i_start, i_end in intervals:
                arr[i_start - start:i_end - start] = True

        return arr

    def pad(self, pad_width, mode='constant', **kwargs):
        """
        Numpy like padding (see np.pad).

        Args:
            pad_width:
                Number of values padded to the edges.
                Either a pair (before, after) or a scalar that is is used for
                before and after.
            mode:
                Only 'constant' is implemented.
            **kwargs:
                Not yet supported, but kept for better error message.

        Returns:
            Padded ArrayInterval


        >>> ai = zeros()
        >>> ai[10:20] = 1
        >>> ai.pad(3)
        ArrayInterval("13:23", shape=None)
        >>> ai.pad([3, 4])
        ArrayInterval("13:23", shape=None)
        >>> ai = zeros(50)
        >>> ai[10:20] = 1
        >>> ai.pad(3)
        ArrayInterval("13:23", shape=(56,))
        >>> ai.pad([3, 4])
        ArrayInterval("13:23", shape=(57,))

        >>> np.pad(np.zeros(50), 3).shape
        (56,)
        >>> np.pad(np.zeros(50), (3, 4)).shape
        (57,)

        """
        if self.inverse_mode:
            raise NotImplementedError(self.inverse_mode)
        if mode != 'constant' or kwargs:
            kwargs = ','.join(
                [f'mode={mode!r} '] + [f'{k}={v!r}' for k, v in
                                       kwargs.items()])
            raise NotImplementedError(kwargs)
        if isinstance(pad_width, int):
            pad_width = [pad_width, pad_width]
        else:
            assert len(pad_width) == 2, pad_width

        shape = self.shape
        shape = shape if shape is None else [*shape[:-1],
                                             shape[-1] + pad_width[0] + pad_width[1]]

        return ArrayInterval.from_pairs([
            [s + pad_width[0], e + pad_width[0]]
            for s, e in self.normalized_intervals
        ], shape, self.inverse_mode)

    def _slice_doctest(self):
        """

        >>> ai = zeros(50)
        >>> ai[10:20] = 1
        >>> ai[25:30] = 1
        >>> ai.slice[19:26]
        ArrayInterval("0:1, 6:7", shape=(7,))

        >>> ai.slice[19:26][:]  # Second getitem converts to np.
        array([ True, False, False, False, False, False,  True])
        >>> ai[19:26]  # Use normal getitem, that returns np.
        array([ True, False, False, False, False, False,  True])

        >>> ai = zeros()
        >>> ai[10:20] = 1
        >>> ai.slice[2:]
        ArrayInterval("8:18", shape=None)
        """
        raise NotImplementedError('Use slice not _slice_doctest.')

    @property
    class slice:
        """
        Similar to __getitem__, but only allow slices as arguments and
        returns an ArrayInterval instead of a numpy.array.

        For examples see _slice_doctest.
        General usage:

            >> new = ai.slice[19:26]

        """
        def __init__(self, ai):
            self.ai = ai

        def __getitem__(self, item):
            sentinel = 9223372036854775807  # 2**63-1
            shape, = [sentinel] if self.ai.shape is None else self.ai.shape

            start, stop = cy_parse_item(item, [shape])
            intervals = cy_intersection((start, stop), self.ai.normalized_intervals)

            if shape == sentinel:
                assert start >= 0, (item, start, stop)
                assert stop >= 0, (item, start, stop)
                if stop == sentinel:
                    shape = None
                else:
                    shape = stop - start
                    assert shape >= 0, (shape, item, start, stop)
            else:
                shape = stop - start
                assert shape >= 0, (shape, item, start, stop)

            return ArrayInterval.from_pairs([
                [s-start, e-start]
                for s, e in intervals
            ], shape, self.ai.inverse_mode)

    def sum(self, axis=None, out=None):
        """
        >>> a = ArrayInterval([True, True, False, False])
        >>> np.sum(a)
        2
        >>> a = ArrayInterval([True, False, False, True])
        >>> np.sum(a)
        2
        >>> np.sum(zeros(10))
        0
        >>> np.sum(ones(10))
        10
        """
        assert out is None, (out, axis, self)
        assert axis is None or axis in (0, -1), (axis, out, self)
        if not self.normalized_intervals:
            sum = 0
        else:
            a, b = np.sum(self.normalized_intervals, axis=0)
            sum = b - a
        if self.inverse_mode:
            sum = self.shape[0] - sum
        return sum

    def mean(self, axis=None, out=None):
        sum = self.sum(axis, out)
        sum /= self.shape[0]
        return sum

    def __or__(self, other):
        """
        >>> a1 = ArrayInterval([True, True, False, False])
        >>> a2 = ArrayInterval([True, False, True, False])
        >>> print(a1 | a2, (a1 | a2)[:])
        ArrayInterval("0:3", shape=(4,)) [ True  True  True False]
        >>> a1 = ArrayInterval([True, True, False, False], inverse_mode=True)
        >>> a2 = ArrayInterval([True, False, True, False], inverse_mode=True)
        >>> print(a1 | a2, (a1 | a2)[:])
        ArrayInterval("3:4", shape=(4,), inverse_mode=True) [ True  True  True False]
        """
        if not isinstance(other, ArrayInterval):
            return NotImplemented
        elif self.inverse_mode is False and other.inverse_mode is False:
            if other.shape != self.shape:
                raise ValueError(
                    f'Cannot broadcast together ArrayIntervals with shapes '
                    f'{self.shape} {other.shape}'
                )
            ai = zeros(shape=self.shape)
            ai.intervals = self.intervals + other.intervals
            return ai
        # elif self.inverse_mode is True and other.inverse_mode is True:
        #     assert other.shape == self.shape, (self.shape, other.shape)
        #     ai = zeros(shape=self.shape)
        #     ai.intervals = self.intervals + other.intervals

        elif self.inverse_mode is True and other.inverse_mode is True:
            return ~((~self) & (~other))
        else:
            raise NotImplementedError(self.inverse_mode, other.inverse_mode)

    def __invert__(self):
        """
        >>> a = ArrayInterval([True, False])
        >>> ~ a
        ArrayInterval("0:1", shape=(2,), inverse_mode=True)
        >>> print(a[:])
        [ True False]
        >>> print((~a)[:])
        [False  True]

        >>> (~ones())[:3]
        array([False, False, False])
        >>> (~zeros())[:3]
        array([ True,  True,  True])
        """
        if self.inverse_mode:
            ai = zeros(shape=self.shape)
        else:
            ai = ones(shape=self.shape)
        ai.intervals = self.intervals
        return ai

    def __and__(self, other):
        """
        >>> a1 = ArrayInterval([True, True, False, False])
        >>> a2 = ArrayInterval([True, False, True, False])
        >>> print(a1 & a2, (a1 & a2)[:])
        ArrayInterval("0:1", shape=(4,)) [ True False False False]
        >>> a1 = ArrayInterval([True, True, False, False], inverse_mode=True)
        >>> a2 = ArrayInterval([True, False, True, False], inverse_mode=True)
        >>> print(a1 & a2, (a1 & a2)[:])
        ArrayInterval("1:4", shape=(4,), inverse_mode=True) [ True False False False]

        >>> np.logical_and(a1, a2)
        array([ True, False, False, False])
        """
        if not isinstance(other, ArrayInterval):
            return NotImplemented
        elif self.inverse_mode is True and other.inverse_mode is True:
            # short circuit
            return ~((~self) | (~other))
        elif self.inverse_mode is False and other.inverse_mode is False:
            if other.shape != self.shape:
                raise ValueError(
                    f'Cannot broadcast together ArrayIntervals with shapes '
                    f'{self.shape} {other.shape}'
                )
            ai = zeros(shape=self.shape)

            normalized_intervals = self.normalized_intervals
            intervals = []
            for (start, stop) in other.normalized_intervals:
                intervals.extend(cy_intersection(
                    (start, stop), normalized_intervals))

            ai.intervals = intervals
            return ai
        else:
            raise NotImplementedError(self.inverse_mode, other.inverse_mode)

    def __xor__(self, other):
        """
        >>> a1 = ArrayInterval([True, True, False, False])
        >>> a2 = ArrayInterval([True, False, True, False])
        >>> print(a1 ^ a2, (a1 ^ a2)[:])
        ArrayInterval("1:3", shape=(4,)) [False  True  True False]
        >>> a1 = ArrayInterval([True, True, False, False], inverse_mode=True)
        >>> a2 = ArrayInterval([True, False, True, False], inverse_mode=True)
        >>> print(a1 ^ a2, (a1 ^ a2)[:])
        ArrayInterval("1:3", shape=(4,)) [False  True  True False]
        """
        if not isinstance(other, ArrayInterval):
            return NotImplemented
        else:
            return _combine(operator.__xor__, self, other)

    def __eq__(self, other):
        """
        >>> a1 = ArrayInterval([True, True, False, False])
        >>> a2 = ArrayInterval([True, False, True, False])
        >>> print(a1 == a2, (a1 == a2)[:])
        ArrayInterval("0:1, 3:4", shape=(4,)) [ True False False  True]
        >>> a1 = ArrayInterval([True, True, False, False], inverse_mode=True)
        >>> a2 = ArrayInterval([True, False, True, False], inverse_mode=True)
        >>> print(a1 == a2, (a1 == a2)[:])
        ArrayInterval("0:1, 3:4", shape=(4,)) [ True False False  True]
        """
        if not isinstance(other, ArrayInterval):
            return NotImplemented
        else:
            return _combine(operator.__eq__, self, other)

    def __ne__(self, other):
        if not isinstance(other, ArrayInterval): return NotImplemented
        else: return _combine(operator.__ne__, self, other)

    def __lt__(self, other):
        if not isinstance(other, ArrayInterval): return NotImplemented
        else: return _combine(operator.__lt__, self, other)

    def __le__(self, other):
        if not isinstance(other, ArrayInterval): return NotImplemented
        else: return _combine(operator.__le__, self, other)

    def __gt__(self, other):
        if not isinstance(other, ArrayInterval): return NotImplemented
        else: return _combine(operator.__gt__, self, other)

    def __ge__(self, other):
        if not isinstance(other, ArrayInterval): return NotImplemented
        else: return _combine(operator.__ge__, self, other)


def _yield_sections(a_intervals, b_intervals):
    """
    >>> a = ArrayInterval._normalize([(0, 2), (6, 8), (20, 30), (33, 35)])
    >>> b = ArrayInterval._normalize([(1, 3), (10, 15), (22, 28), (35, 37)])
    >>> a
    ((0, 2), (6, 8), (20, 30), (33, 35))
    >>> b
    ((1, 3), (10, 15), (22, 28), (35, 37))
    >>> for s in _yield_sections(a, b):
    ...     print(s)
    (0, 1, True, False)
    (1, 2, True, True)
    (2, 3, False, True)
    (3, 6, False, False)
    (6, 8, True, False)
    (8, 10, False, False)
    (10, 15, False, True)
    (15, 20, False, False)
    (20, 22, True, False)
    (22, 28, True, True)
    (28, 30, True, False)
    (30, 33, False, False)
    (33, 35, True, False)
    (35, 37, False, True)

    >>> c = ArrayInterval._normalize([
    ...     (start, stop)
    ...     for start, stop, a_, b_ in _yield_sections(a, b)
    ...     if a_ ^ b_
    ... ])
    >>> c
    ((0, 1), (2, 3), (6, 8), (10, 15), (20, 22), (28, 30), (33, 37))
    """

    a_intervals_iter = iter(a_intervals)
    b_intervals_iter = iter(b_intervals)

    a_start, a_end = next(a_intervals_iter)
    b_start, b_end = next(b_intervals_iter)

    current_position = 0

    # while True:
    for _ in range(10 * (len(a_intervals) + len(b_intervals))):
        if a_start > current_position and b_start > current_position:
            new_pos = min(a_start, b_start)
            yield (current_position, new_pos, False, False)
            current_position = new_pos

        elif a_start <= current_position and b_start > current_position:
            new_pos = min(a_end, b_start)
            yield (current_position, new_pos, True, False)
            current_position = new_pos

        elif a_start > current_position and b_start <= current_position:
            new_pos = min(b_end, a_start)
            yield (current_position, new_pos, False, True)
            current_position = new_pos

        elif a_start == current_position or b_start == current_position:
            new_pos = min(a_end, b_end)
            yield (current_position, new_pos, True, True)
            current_position = new_pos

        else:
            raise RuntimeError(current_position, a_start, a_end, b_start, b_end)

        if current_position == a_end:
            try:
                a_start, a_end = next(a_intervals_iter)
            except StopIteration:
                a_start = float('inf')

        if current_position == b_end:
            try:
                b_start, b_end = next(b_intervals_iter)
            except StopIteration:
                b_start = float('inf')

        if current_position == max(a_end, b_end):
            break


def _combine(func, *array_intervals, out=None):
    """

    >>> import operator
    >>> ai1 = ArrayInterval(np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=bool))
    >>> ai1
    ArrayInterval("3:5, 8:10", shape=(11,))
    >>> ai2 = ArrayInterval(np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=bool))
    >>> ai2
    ArrayInterval("6:10", shape=(11,))
    >>> _combine(operator.__or__, ai1, ai2)
    ArrayInterval("3:5, 6:10", shape=(11,))
    >>> _combine(operator.__and__, ai1, ai2)
    ArrayInterval("8:10", shape=(11,))
    >>> _combine(operator.__xor__, ai1, ai2)
    ArrayInterval("3:5, 6:8", shape=(11,))
    >>> _combine(operator.__not__, ai1)
    ArrayInterval("0:3, 5:8, 10:11", shape=(11,))

    >>> ai1.shape = None
    >>> ai2.shape = None
    >>> ai1
    ArrayInterval("3:5, 8:10", shape=None)
    >>> ai2
    ArrayInterval("6:10", shape=None)
    >>> _combine(operator.__or__, ai1, ai2)
    ArrayInterval("3:5, 6:10", shape=None)
    >>> _combine(operator.__and__, ai1, ai2)
    ArrayInterval("8:10", shape=None)
    >>> _combine(operator.__xor__, ai1, ai2)
    ArrayInterval("3:5, 6:8", shape=None)
    >>> _combine(operator.__not__, ai1)
    ArrayInterval("3:5, 8:10", shape=None, inverse_mode=True)
    >>> _combine(operator.__not__, ai1)[:11]
    array([ True,  True,  True, False, False,  True,  True,  True, False,
           False,  True])

    >>> _combine(operator.__or__, ai1, ArrayInterval(ai1[:11]))
    ArrayInterval("3:5, 8:10", shape=(11,))
    >>> _combine(operator.__or__, ai1, ArrayInterval(ai1[:10]))
    ArrayInterval("3:5, 8:10", shape=(10,))
    >>> _combine(operator.__or__, ai1, ArrayInterval(ai1[:9]))
    Traceback (most recent call last):
    ...
    IndexError: Index 9 is out of bounds for ArrayInterval with shape (9,)

    """

    edges = {0, }
    for ai in array_intervals:
        ai: ArrayInterval
        for start_end in ai.normalized_intervals:
            edges.update(start_end)

    edges = sorted(edges)

    values = [ai.inverse_mode for ai in array_intervals]
    last = func(*values)

    if out is None:
        shapes = [ai.shape for ai in array_intervals if ai.shape is not None]
        assert len(set(shapes)) in [0, 1], shapes
        # assert len(set(shapes)) == 1, shapes
        shape = shapes[0] if shapes else None

        if shape is None:
            if last:
                out = ones(shape=shape)
            else:
                out = zeros(shape=shape)
        else:
            out = zeros(shape=shape)
            out[edges[-1]:] = last
    else:
        out: ArrayInterval
        if out.shape is None:
            assert last == out.inverse_mode, (last, func, values, out,)
        else:
            out[edges[-1]:] = last

    for s, e in zip(edges, edges[1:]):
        values = [ai[s] for ai in array_intervals]
        # print(s, e, values, func(*values))
        out[s:e] = func(*[ai[s] for ai in array_intervals])
    return out
