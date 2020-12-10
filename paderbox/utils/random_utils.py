import numpy as np
from functools import wraps
from scipy.stats import truncnorm, truncexpon
import dataclasses

__all__ = [
    'str_to_random_state',
    'uniform',
    'log_uniform',
    'randn',
    'normal',
    'truncated_normal',
    'log_truncated_normal',
    'truncated_exponential',
    'hermitian',
    'pos_def_hermitian',
    'Uniform',
    'LogUniform',
    'TruncatedNormal',
    'LogTruncatedNormal',
    'TruncatedExponential',
]


def str_to_random_state(string):
    """
    This functions outputs a consistent random state dependent on
    an input string.
    
    >>> print(str_to_random_state(''))
    RandomState(MT19937)
    """
    import hashlib
    hash_value = hashlib.sha256(string.encode("utf-8"))
    hash_value = int(hash_value.hexdigest(), 16)
    hash_value = hash_value % 2 ** 32
    return np.random.RandomState(hash_value)


def _force_correct_shape(f):
    """decorator allowing to pass shape as tuple as well

    Args:
        f: wrapped function

    Returns:

    """
    @wraps(f)
    def wrapper(*shape, **kwargs):
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            assert len(shape) == 1, shape
            shape = shape[0]

        return f(*shape, **kwargs)

    return wrapper


def _add_kwarg_dtype(default):
    """ returns decorator adding the kwarg dtype to the wrapped function and
    handles the sampling of a certain dtype.

    Args:
        default: default dtype

    Returns:

    """
    def _decorator(f):
        @wraps(f)
        def wrapper(*shape, dtype=default, **kwargs):

            def _f(dtype_local):
                return np.array(f(*shape, **kwargs), dtype=dtype_local)

            if dtype in (np.float32, np.float64):
                return _f(dtype)
            elif dtype is np.complex64:
                return _f(np.float32) + 1j * _f(np.float32)
            elif dtype is np.complex128:
                return _f(np.float64) + 1j * _f(np.float64)
            else:
                raise ValueError(f'Invalid dtype {dtype}')

        return wrapper
    return _decorator


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def uniform(*shape, low=-1., high=1.):
    """

    Args:
        *shape:
        low:
        high:
        dtype:

    Returns:

    >>> x = uniform()
    >>> x.ndim
    0
    >>> x.dtype
    dtype('float64')
    >>> x = uniform(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = uniform(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return np.random.uniform(low=low, high=high, size=shape)


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def log_uniform(*shape, low=-1., high=1.):
    """

    Args:
        *shape:
        low:
        high:
        dtype:

    Returns:

    >>> x = log_uniform()
    >>> x.ndim
    0
    >>> x = log_uniform(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = log_uniform(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return np.exp(np.random.uniform(low=low, high=high, size=shape))


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def randn(*shape):
    """

    Args:
        *shape:
        dtype:

    Returns:

    >>> x = randn()
    >>> x.ndim
    0
    >>> x = randn(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = randn(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return np.random.randn(*shape)


def normal(*shape, loc=0., scale=1., dtype=np.complex128):
    """

    Args:
        *shape:
        loc:
        scale:
        dtype:

    Returns:

    """
    return scale * randn(*shape, dtype=dtype) + loc


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def truncated_normal(*shape, loc=0., scale=1., truncation=3.):
    """samples from normal distribution with high deviations being truncated

    Args:
        *shape:
        loc:
        scale:
        truncation: max deviation from loc beyond which the distribution is
            truncated. E.g., with a loc of 1 and truncation of 3 the
            distribution is truncated at -2 and +4.
        dtype:

    Returns:

    >>> x = truncated_normal()
    >>> x.ndim
    0
    >>> x = truncated_normal(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = truncated_normal(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return (
        truncnorm(-truncation / scale, truncation / scale, loc, scale).rvs(shape)
    )


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def log_truncated_normal(*shape, loc=0., scale=.5, truncation=3.):
    """exp(.) of truncated-normal distributed random variables

    Args:
        *shape:
        loc:
        scale:
        truncation: max deviation from loc beyond which the normal distribution
            (in log domain) is truncated. E.g., with a loc of 1 and truncation
            of 3 the normal distribution (in log-domain) is truncated at -2
            and +4 and hence the log-normal distribution is truncated at
            exp(-2) and exp(+4).
        dtype:

    Returns:

    >>> x = log_truncated_normal()
    >>> x.ndim
    0
    >>> x = log_truncated_normal(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = log_truncated_normal(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return np.exp(truncnorm(-truncation / scale, truncation / scale, loc, scale).rvs(shape))


@_force_correct_shape
@_add_kwarg_dtype(default=np.float64)
def truncated_exponential(*shape, loc=0., scale=1., truncation=3.):
    """

    Args:
        *shape:
        loc:
        scale:
        truncation: max deviation from loc beyond which the distribution
            is truncated. E.g., with a loc of 1  and truncation of 3 the
            distribution is truncated at +4.
        dtype:

    Returns:

    >>> x = truncated_exponential()
    >>> x.ndim
    0
    >>> x = truncated_exponential(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = truncated_exponential(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    """
    return truncexpon(truncation / scale, loc, scale).rvs(shape)


@_force_correct_shape
def hermitian(*shape, dtype=np.complex128):
    """ Assures a random positive-semidefinite hermitian matrix.

    Args:
        *shape:
        dtype:

    Returns:

    """
    assert len(shape) >= 2 and shape[-1] == shape[-2], shape
    matrix = uniform(*shape, dtype=dtype)
    matrix = matrix + np.swapaxes(matrix, -1, -2).conj()
    np.testing.assert_allclose(matrix, np.swapaxes(matrix, -1, -2).conj())
    return matrix


@_force_correct_shape
def pos_def_hermitian(*shape, dtype=np.complex128):
    """Assures a random POSITIVE-DEFINITE hermitian matrix.

    TODO: Can this be changed? Why do we need 2?

    Args:
        *shape:
        dtype:

    Returns:

    """
    matrix = hermitian(*shape, dtype=dtype)
    matrix += np.broadcast_to(shape[-1] * 2 * np.eye(shape[-1]), shape)
    return matrix


@dataclasses.dataclass
class Uniform:
    low: float = -1.
    high: float = 1.
    dtype: type = np.float64

    def __call__(self, *shape):
        return uniform(*shape, low=self.low, high=self.high, dtype=self.dtype)


@dataclasses.dataclass
class LogUniform:
    low: float = -1.
    high: float = 1.
    dtype: type = np.float64

    def __call__(self, *shape):
        return log_uniform(
            *shape, low=self.low, high=self.high, dtype=self.dtype
        )


@dataclasses.dataclass
class TruncatedNormal:
    loc: float = 0.
    scale: float = 1.
    truncation: float = 3.
    dtype: type = np.float64

    def __call__(self, *shape):
        return truncated_normal(
            *shape, loc=self.loc, scale=self.scale, truncation=self.truncation,
            dtype=self.dtype
        )


@dataclasses.dataclass
class LogTruncatedNormal:
    loc: float = 0.
    scale: float = 1.
    truncation: float = 3.
    dtype: type = np.float64

    def __call__(self, *shape):
        return log_truncated_normal(
            *shape, loc=self.loc, scale=self.scale, truncation=self.truncation,
            dtype=self.dtype
        )


@dataclasses.dataclass
class TruncatedExponential:
    loc: float = 0.
    scale: float = 1.
    truncation: float = 3.
    dtype: type = np.float64

    def __call__(self, *shape):
        return truncated_exponential(
            *shape, loc=self.loc, scale=self.scale, truncation=self.truncation,
            dtype=self.dtype
        )
