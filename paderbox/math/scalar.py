import numpy as np


def abs_square(x: np.ndarray):
    """

    https://github.com/numpy/numpy/issues/9679

    Bug in numpy 1.13.1
    >> np.ones(32768).imag ** 2
    Traceback (most recent call last):
    ...
    ValueError: output array is read-only
    >> np.ones(32767).imag ** 2
    array([ 0.,  0.,  0., ...,  0.,  0.,  0.])

    >>> abs_square(np.ones(32768)).shape
    (32768,)
    >>> abs_square(np.ones(32768, dtype=np.complex64)).shape
    (32768,)
    """

    if np.iscomplexobj(x):
        return x.real ** 2 + x.imag ** 2
    else:
        return x ** 2