"""
`ArrayInterval` offers a user-readable object for stashing activity information.
In combination with jsonpickle this allows for a low resource possibility to
save activity information for large time streams.
"""

from typing import Optional, Union, Iterable

import numpy as np
from paderbox.array.interval.util import (
    cy_non_intersection,
    cy_intersection,
    cy_parse_item,
    cy_str_to_intervals,
)


def ArrayInterval_from_str(string, shape):
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
        print('empty interval found')
        pass
    else:
        if not ',' in string:
            string = string + ','
        try:
            ai.add_intervals_from_str(string)
        except Exception as e:
            raise Exception(string) from e
    return ai


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

    if isinstance(shape, int):
        shape = [shape]

    if shape is not None:
        assert len(shape) == 1, shape
        shape = tuple(shape)

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

    if isinstance(shape, int):
        shape = [shape]

    if shape is not None:
        assert len(shape) == 1, shape
        shape = tuple(shape)

    ai.shape = shape
    return ai


class ArrayInterval:
    from_str = staticmethod(ArrayInterval_from_str)
    inverse_mode = False

    def __init__(self, array, inverse_mode: bool = False):
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
                This flag is nessesary when the shape is unknown.
                The user does not need to care about this flag. The default is
                fine.

        Examples:
            >>> ai = ArrayInterval(np.array([1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.bool))
            >>> ai
            ArrayInterval("0:2, 3:4, 6:8", shape=(9,))
            >>> ai[:]
            array([ True,  True, False,  True, False, False,  True,  True, False])
            >>> a = np.array([1, 1, 1, 1], dtype=np.bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([0, 0, 0, 0], dtype=np.bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([0, 1, 1, 0], dtype=np.bool)
            >>> assert all(a == ArrayInterval(a)[:])
            >>> a = np.array([1, 0, 0, 1], dtype=np.bool)
            >>> assert all(a == ArrayInterval(a)[:])

        """
        if isinstance(array, ArrayInterval):
            self.shape = array.shape
            self.inverse_mode = array.inverse_mode
            self.intervals = array.intervals
        else:
            array = np.asarray(array)
            assert array.ndim == 1, (array.ndim, array)
            assert array.dtype == np.bool, (np.bool, array)

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
            self.shape = array.shape

            if inverse_mode:
                for start, stop in zip(rising, falling):
                    self[start:stop] = 0
            else:
                for start, stop in zip(rising, falling):
                    self[start:stop] = 1

    def __copy__(self):
        if self.inverse_mode:
            ai = ones(shape=self.shape)
        else:
            ai = zeros(shape=self.shape)
        ai.intervals = self.intervals
        return ai

    def __array__(self, dtype=np.bool):
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
            RuntimeError: You cannot cast an ArrayInterval to numpy,
            when the shape is unknown.
        """
        assert dtype == np.bool, dtype
        if self.shape is None:
            raise RuntimeError(
                f'You cannot cast an {self.__class__.__name__} to numpy,\n'
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
          {'py/tuple': ['1:4, 5:20, 21:25', 50]}]}
        """
        return self.from_str, (self._intervals_as_str, self.shape[-1])

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
        i_str = []
        for i in self.normalized_intervals:
            start, end = i
            #             i_str += [f'[{start}, {end})']
            i_str += [f'{start}:{end}']

        i_str = ', '.join(i_str)
        return i_str

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

        """

        start, stop = cy_parse_item(item, self.shape)

        if np.isscalar(value):
            if self.inverse_mode:
                if value == 0:
                    self.intervals = self.intervals + ((start, stop),)
                elif value == 1:
                    self.intervals = cy_non_intersection((start, stop), self.intervals)
                else:
                    raise ValueError(value)
            else:
                if value == 1:
                    self.intervals = self.intervals + ((start, stop),)
                elif value == 0:
                    self.intervals = cy_non_intersection((start, stop), self.intervals)
                else:
                    raise ValueError(value)
        elif isinstance(value, (tuple, list, np.ndarray)):
            assert len(value) == stop - start, (start, stop, stop - start, len(value), value)
            ai = ArrayInterval(value, inverse_mode=self.inverse_mode)
            intervals = self.intervals
            intervals = cy_non_intersection((start, stop), intervals)
            self.intervals = intervals + tuple([(s + start, e + start) for s, e in ai.intervals])
        else:
            raise NotImplementedError(value)

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
     
        """
        if isinstance(item, (int, np.integer)):
            # Could be optimized
            for s, e in self.normalized_intervals:
                if e > item:
                    return (item >= s) ^ self.inverse_mode
            return self.inverse_mode

        start, stop = cy_parse_item(item, self.shape)
        intervals = cy_intersection((start, stop), self.normalized_intervals)

        if self.inverse_mode:
            arr = np.ones(stop - start, dtype=np.bool)

            for i_start, i_end in intervals:
                arr[i_start - start:i_end - start] = False
        else:
            arr = np.zeros(stop - start, dtype=np.bool)

            for i_start, i_end in intervals:
                arr[i_start - start:i_end - start] = True

        return arr

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
        assert axis is None, (axis, out, self)
        if not self.normalized_intervals:
            sum = 0
        else:
            a, b = np.sum(self.normalized_intervals, axis=0)
            sum = b - a
        if self.inverse_mode:
            sum = self.shape[0] - sum
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
            assert other.shape == self.shape, (self.shape, other.shape)
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
            assert other.shape == self.shape, (self.shape, other.shape)
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
            import operator
            return _combine(operator.__xor__, self, other)


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

    """

    edges = {0, }
    for ai in array_intervals:
        ai: ArrayInterval
        for start_end in ai.normalized_intervals:
            edges.update(start_end)

    edges = sorted(edges)

    values = [ai[edges[-1]] for ai in array_intervals]
    last = func(*values)

    if out is None:
        shapes = [ai.shape for ai in array_intervals]
        assert len(set(shapes)) == 1, shapes
        shape = shapes[0]

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
