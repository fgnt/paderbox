import collections
import itertools
import operator
from collections.abc import Mapping

from typing import Callable, Any, Hashable, Optional, Union, Iterable

__all__ = [
    'groupby',
    'zip',
]


def zip(*iterables, strict=False):
    """
    Python implementation of strict zip, introduced in Python 3.10.
    https://peps.python.org/pep-0618/
    """
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        if not strict:
            return
    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i+1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i+1} is longer than argument{plural}{i}"
            raise ValueError(msg)


def groupby(
        iterable,
        key: Optional[Union[Callable[[Any], Hashable], str, int, Iterable]] = None,
):
    """
    A non-lazy variant of `itertools.groupby` with advanced features.

    Args:
        iterable: Iterable to group
        key: Determines by what to group. Can be:
            - `None`: Use the iterables elements as keys directly
            - `callable`: Gets called with every element and returns the group
                key
            - `str`, or `int`: Use `__getitem__` on elements in `iterable`
                to obtain the key
            - `Iterable`: Provides the keys. Has to have the same length as
                `iterable`.

    Examples:
        >>> groupby('ab'*3)
        {'a': ['a', 'a', 'a'], 'b': ['b', 'b', 'b']}
        >>> groupby(range(10), lambda x: x%2)
        {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
        >>> groupby(({'a': x%2, 'b': x} for x in range(3)), 'a')
        {0: [{'a': 0, 'b': 0}, {'a': 0, 'b': 2}], 1: [{'a': 1, 'b': 1}]}
        >>> groupby(['abc', 'bd', 'abd', 'cdef', 'c'], 0)
        {'a': ['abc', 'abd'], 'b': ['bd'], 'c': ['cdef', 'c']}
        >>> groupby(range(10), list(range(5))*2)
        {0: [0, 5], 1: [1, 6], 2: [2, 7], 3: [3, 8], 4: [4, 9]}
        >>> groupby('abc', ['a'])
        Traceback (most recent call last):
            ...
        ValueError: zip() argument 2 is shorter than argument 1
        >>> groupby('abc', {})
        Traceback (most recent call last):
            ...
        TypeError: Invalid type for key: <class 'dict'>
    """
    if callable(key) or key is None:
        key_fn = key
    elif isinstance(key, (str, int)):
        key_fn = operator.itemgetter(key)
    elif not isinstance(key, Mapping):
        value_getter = operator.itemgetter(0)
        groups = collections.defaultdict(list)
        for key, group in itertools.groupby(zip(iterable, key, strict=True), operator.itemgetter(1)):
            groups[key].extend(map(value_getter, group))
        return dict(groups)
    else:
        raise TypeError(f'Invalid type for key: {type(key)}')

    groups = collections.defaultdict(list)
    for key, group in itertools.groupby(iterable, key_fn):
        groups[key].extend(group)
    return dict(groups)
