""" A collection of (mostly undocumented) functions used in various projects.

Most of these functions are tailored for a specific need. However, you might
find something useful if you alter it a little bit.

"""

import collections
import os
import sys


def update_dict(d, u):
    """ Recursively update dict d with values from dict u.

    Args:
        d: Dict to be updated
        u: Dict with values to use for update

    Returns: Updated dict

    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            default = v.copy()
            default.clear()
            r = update_dict(d._get(k, default), v)
            d[k] = r
        else:
            d[k] = v
    return d


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
