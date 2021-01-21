import numpy as np
from scipy.stats import truncnorm, truncexpon
import dataclasses
from typing import Union

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
    'Normal',
    'TruncatedNormal',
    'LogTruncatedNormal',
    'TruncatedExponential',
]


def str_to_seed(string: str, bits: int = 32) -> int:
    """
    This functions outputs a consistent seed in the range (0, 2**bits - 1)
    dependent on an input string and the number of `bits`.

    >>> print(str_to_seed(''))
    2018687061
    """
    import hashlib
    hash_value = hashlib.sha256(string.encode("utf-8"))
    hash_value = int(hash_value.hexdigest(), 16)
    hash_value = hash_value % 2 ** bits
    return hash_value


def str_to_random_state(
        string: str,
        seed_bits: int = 32
) -> 'np.random.RandomState':
    """
    This functions outputs a consistent random state (`np.random.RandomState`)
    dependent on an input string and the number of bits (`seed_bits`).
    
    >>> print(str_to_random_state(''))
    RandomState(MT19937)

    Notes:
        It is not recommended to use a seed with less than 32 bits
        (https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy).
        `RandomState` doesn't support seeds larger than 2**32 - 1. So, the value
        of `seed_bits` shouldn't be changed.
    """
    return np.random.RandomState(str_to_seed(string, bits=seed_bits))


def str_to_random_generator(
        string: str,
        seed_bits: int = 128
) -> 'np.random.Generator':
    """
    This functions outputs a consistent random number generator
    (`np.random.Generator`) dependent on an input string and the number of bits
    (`seed_bits`).

    >>> print(str_to_random_generator(''))
    Generator(PCG64)

    Notes:
        The `seed_bits` is set to 128 here, which is the default for the PCG64
        generator. It is not recommended to use less than 32 bits for the seed.
        See https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy.
    """
    return np.random.default_rng(str_to_seed(string, bits=seed_bits))


def _force_correct_shape(shape):
    if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
        assert len(shape) == 1, shape
        shape = shape[0]
    return shape


@dataclasses.dataclass
class _Sampler:
    dtype: Union[type, str] = 'float64'

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = getattr(np, self.dtype)
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype

    def _sample(self, shape):
        raise NotImplementedError

    def __call__(self, *shape):
        shape = _force_correct_shape(shape)

        def _f(dtype_local):
            return np.array(self._sample(shape), dtype=dtype_local)

        if self.dtype in (np.float32, np.float64):
            return _f(self.dtype)
        elif self.dtype is np.complex64:
            return _f(np.float32) + 1j * _f(np.float32)
        elif self.dtype is np.complex128:
            return _f(np.float64) + 1j * _f(np.float64)
        else:
            raise ValueError(f'Invalid dtype {self.dtype}')


@dataclasses.dataclass
class Uniform(_Sampler):
    low: float = 0.
    high: float = 1.

    def __post_init__(self):
        super().__post_init__()
        assert self.low < self.high, (self.low, self.high)

    def _sample(self, shape):
        return np.random.uniform(low=self.low, high=self.high, size=shape)


@dataclasses.dataclass
class LogUniform(Uniform):
    def _sample(self, shape):
        return np.exp(super()._sample(shape))


@dataclasses.dataclass
class Normal(_Sampler):
    loc: float = 0.
    scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        assert self.scale > 0, self.scale

    def _sample(self, shape):
        return self.scale * np.random.randn(*shape) + self.loc


@dataclasses.dataclass
class TruncatedNormal(Normal):
    truncation: float = 3.

    def _sample(self, shape):
        return truncnorm(
            -self.truncation / self.scale,
            self.truncation / self.scale, self.loc,
            self.scale
        ).rvs(shape)


@dataclasses.dataclass
class LogTruncatedNormal(TruncatedNormal):
    def _sample(self, shape):
        return np.exp(super()._sample(shape))


@dataclasses.dataclass
class TruncatedExponential(_Sampler):
    loc: float = 0.
    scale: float = 1.
    truncation: float = 3.

    def __post_init__(self):
        super().__post_init__()
        assert self.scale > 0, self.scale

    def _sample(self, shape):
        return truncexpon(self.truncation / self.scale, self.loc, self.scale).rvs(shape)


def uniform(*shape, low=0., high=1., dtype=np.float64):
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
    >>> x = uniform(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = uniform(2, 3, low=2.)
    Traceback (most recent call last):
        ...
        assert self.low < self.high, (self.low, self.high)
    AssertionError: (2.0, 1.0)
    """
    return Uniform(low=low, high=high, dtype=dtype)(*shape)


def log_uniform(*shape, low=0., high=1., dtype=np.float64):
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
    >>> x = log_uniform(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = log_uniform(2, 3, low=2.)
    Traceback (most recent call last):
        ...
        assert self.low < self.high, (self.low, self.high)
    AssertionError: (2.0, 1.0)
    """
    return LogUniform(low=low, high=high, dtype=dtype)(*shape)


def randn(*shape, dtype=np.float64):
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
    >>> x = randn(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    """
    return Normal(dtype=dtype)(*shape)


def normal(*shape, loc=0., scale=1., dtype=np.float64):
    """

    Args:
        *shape:
        loc:
        scale:
        dtype:

    Returns:

    >>> x = normal()
    >>> x.ndim
    0
    >>> x = normal(2, 3)
    >>> x.shape, x.dtype
    ((2, 3), dtype('float64'))
    >>> x = normal(2, 3, dtype=np.complex128)
    >>> x.shape, x.dtype
    ((2, 3), dtype('complex128'))
    >>> x = normal(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = normal(2, 3, scale=-1)
    Traceback (most recent call last):
        ...
        assert self.scale > 0, self.scale
    AssertionError: -1

    """
    return Normal(loc=loc, scale=scale, dtype=dtype)(*shape)


def truncated_normal(*shape, loc=0., scale=1., truncation=3., dtype=np.float64):
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
    >>> x = truncated_normal(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = truncated_normal(2, 3, scale=-1)
    Traceback (most recent call last):
        ...
        assert self.scale > 0, self.scale
    AssertionError: -1

    """
    return TruncatedNormal(loc=loc, scale=scale, truncation=truncation, dtype=dtype)(*shape)


def log_truncated_normal(*shape, loc=0., scale=.5, truncation=3., dtype=np.float64):
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
    >>> x = log_truncated_normal(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = log_truncated_normal(2, 3, scale=-1)
    Traceback (most recent call last):
        ...
        assert self.scale > 0, self.scale
    AssertionError: -1

    """
    return LogTruncatedNormal(loc=loc, scale=scale, truncation=truncation, dtype=dtype)(*shape)


def truncated_exponential(*shape, loc=0., scale=1., truncation=3., dtype=np.float64):
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
    >>> x = truncated_exponential(2, 3, dtype=np.int64)
    Traceback (most recent call last):
        ...
        assert self.dtype in (np.float32, np.float64, np.complex64, np.complex128), self.dtype
    AssertionError: <class 'numpy.int64'>
    >>> x = truncated_exponential(2, 3, scale=-1)
    Traceback (most recent call last):
        ...
        assert self.scale > 0, self.scale
    AssertionError: -1

    """
    return TruncatedExponential(loc=loc, scale=scale, truncation=truncation, dtype=dtype)(*shape)


def hermitian(*shape, dtype=np.complex128):
    """ Assures a random positive-semidefinite hermitian matrix.

    Args:
        *shape:
        dtype:

    Returns:

    """

    shape = _force_correct_shape(shape)
    assert len(shape) >= 2 and shape[-1] == shape[-2], shape
    matrix = uniform(*shape, dtype=dtype)
    matrix = matrix + np.swapaxes(matrix, -1, -2).conj()
    np.testing.assert_allclose(matrix, np.swapaxes(matrix, -1, -2).conj())
    return matrix


def pos_def_hermitian(*shape, dtype=np.complex128):
    """Assures a random POSITIVE-DEFINITE hermitian matrix.

    TODO: Can this be changed? Why do we need 2?

    Args:
        *shape:
        dtype:

    Returns:

    """
    shape = _force_correct_shape(shape)
    matrix = hermitian(*shape, dtype=dtype)
    matrix += np.broadcast_to(shape[-1] * 2 * np.eye(shape[-1]), shape)
    return matrix
