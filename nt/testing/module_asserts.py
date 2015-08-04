"""
This file contains the STFT function and related helper functions.
"""

from numpy.testing.utils import assert_array_compare
import operator


"""
This is a copy of numpy.testing.assert_array_less.

Raises an AssertionError if two array_like objects are not ordered by less
than.

Given two array_like objects, check that the shape is equal and all
elements of the first object are strictly smaller than those of the
second object. An exception is raised at shape mismatch or incorrectly
ordered values. Shape mismatch does not raise if an object has zero
dimension. In contrast to the standard usage in numpy, NaNs are
compared, no assertion is raised if both objects have NaNs in the same
positions.



Parameters
----------
x : array_like
  The smaller object to check.
y : array_like
  The larger object to compare.
err_msg : string
  The error message to be printed in case of failure.
verbose : bool
    If True, the conflicting values are appended to the error message.

Raises
------
AssertionError
  If actual and desired objects are not equal.

See Also
--------
assert_array_equal: tests objects for equality
assert_array_almost_equal: test objects for equality up to precision



Examples
--------
>>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1.1, 2.0, np.nan])
>>> np.testing.assert_array_less([1.0, 1.0, np.nan], [1, 2.0, np.nan])
...
<type 'exceptions.ValueError'>:
Arrays are not less-ordered
(mismatch 50.0%)
 x: array([  1.,   1.,  NaN])
 y: array([  1.,   2.,  NaN])

>>> np.testing.assert_array_less([1.0, 4.0], 3)
...
<type 'exceptions.ValueError'>:
Arrays are not less-ordered
(mismatch 50.0%)
 x: array([ 1.,  4.])
 y: array(3)

>>> np.testing.assert_array_less([1.0, 2.0, 3.0], [4])
...
<type 'exceptions.ValueError'>:
Arrays are not less-ordered
(shapes (3,), (1,) mismatch)
 x: array([ 1.,  2.,  3.])
 y: array([4])

"""


def assert_array_greater(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__gt__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_array_greater_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__ge__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_array_less_equal(x, y, err_msg='', verbose=True):
    assert_array_compare(operator.__le__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not greater-ordered')


def assert_isreal(actual, err_msg='', verbose=True):
    """
    Raises an AssertionError if object is not real.

    The test is equivalent to ``isreal(actual)``.

    Parameters
    ----------
    actual : array_like
        Array obtained.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual is not real.

    See Also
    --------
    assert_allclose
    """

    import numpy as np
    np.testing.assert_equal(np.isreal(actual), True, err_msg, verbose)
