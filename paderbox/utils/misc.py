""" A collection of (mostly undocumented) functions used in various projects.

Most of these functions are tailored for a specific need. However, you might
find something useful if you alter it a little bit.

"""

import os
import sys
from collections.abc import Mapping
from typing import Iterable, Hashable


def interleave(*lists):
    """ Interleave multiple lists. Input does not need to be of equal length.

    based on http://stackoverflow.com/a/29566946/911441

    >>> a = [1, 2, 3, 4, 5]
    >>> b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    >>> list(interleave(a, b))
    [1, 'a', 2, 'b', 3, 'c', 4, 'd', 5, 'e', 'f', 'g']
    >>> list(interleave(b, a))
    ['a', 1, 'b', 2, 'c', 3, 'd', 4, 'e', 5, 'f', 'g']

    Args:
        lists: An arbitrary number of lists

    Returns: Interleaved lists

    """
    iterators = [iter(l) for l in lists]
    while True:
        for iter_idx in range(len(iterators)):
            try:
                if iterators[iter_idx] is not None:
                    yield next(iterators[iter_idx])
            except StopIteration:
                iterators[iter_idx] = None
        if all(i is None for i in iterators):
            return


class PrintSuppressor:
    """Context manager to suppress print output.

    Source: https://stackoverflow.com/a/45669280
    """
    # pylint: disable=attribute-defined-outside-init
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def all_equal(x: Iterable[Hashable]) -> bool:
    """
    Checks if all elements in `x` are equal. Returns `False` if `x` is empty.

    Defined to improve readability.

    Examples:
        >>> all_equal((1, 1, 1, 1))
        True
        >>> all_equal((1, 1, 1, 2))
        False
    """
    if isinstance(x, Mapping):
        raise TypeError('all_equal does not support Mappings')
    return len(set(x)) == 1


def all_unique(x: Iterable[Hashable]) -> bool:
    """
    Checks if all elements in `x` are unique. Returns `True` if `x` is empty.

    Defined to improve readability.

    Examples:
        >>> all_unique((1, 2, 3, 4))
        True
        >>> all_unique((1, 2, 3, 1))
        False
    """
    if isinstance(x, Mapping):
        raise TypeError('all_unique does not support Mappings')
    return len(set(x)) == len(list(x))


def all_in(x: Iterable[Hashable], y: Iterable[Hashable]) -> bool:
    """
    Check if all elements in `x` are in `y`. Returns `True` if `x` is empty.

    Equivalent to `set(x).issubset(y)`.

    Defined to improve readability.

    Examples:
        >>> all_in([1, 2, 2, 1, 2], [1, 2, 3])
        True
        >>> all_in([1, 2, 2, 4, 2], [1, 2, 3])
        False
    """
    return set(x).issubset(y)


def any_in(x: Iterable[Hashable], y: Iterable[Hashable]) -> bool:
    """
    Check if any elements in `x` is in `y`. Returns `True` if `x` is empty.

    Defined to improve readability.

    Examples:
        >>> any_in([1, 2, 2, 1, 2], [1, 2, 3])
        True
        >>> any_in([1, 2, 2, 4, 2], [1, 2, 3])
        True
        >>> any_in([1, 2, 2, 4, 2], [3, 5, 6])
        False
    """
    return bool(set(x).intersection(y))
