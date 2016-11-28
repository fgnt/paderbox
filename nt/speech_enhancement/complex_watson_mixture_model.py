import functools
import math

import numpy as np
from nt.speech_enhancement.beamformer import get_pca
from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from nt.transform import get_stft_center_frequencies
from nt.utils.numpy_utils import reshape
from nt.TODO import bss
from scipy.interpolate import interp1d
from scipy.special import hyp1f1


def _is_power_of_two(number):
    """
    >>> _is_power_of_two(0)
    False
    >>> _is_power_of_two(1)
    True
    >>> _is_power_of_two(2)
    True
    """
    return not (number == 0) and not number & (number - 1)


def normalize_observation(
        signal, unit_norm=True, phase_norm=False, frequency_norm=False,
        max_sensor_distance=None, shrink_factor=1.2,
        fft_size=1024, sample_rate=16000, sound_velocity=343
):
    """ Different feature normalizations.

    Args:
        signal: STFT signal with shape (..., sensors, frequency, time).
        phase_norm: The first sensor element will be real and positive.
        unit_norm: Normalize vector length to length one.
        frequency_norm:
        max_sensor_distance:
        shrink_factor:
        fft_size:
        sample_rate:
        sound_velocity:

    Returns:

    """
    D, F, T = signal.shape[-3:]
    assert _is_power_of_two(F - 1)

    signal = np.copy(signal)

    if unit_norm:
        signal /= np.linalg.norm(signal, axis=-3, keepdims=True)

    if phase_norm:
        signal *= np.exp(-1j * np.angle(signal[..., 0, :, :]))

    if frequency_norm:
        frequency = get_stft_center_frequencies(fft_size, sample_rate)
        assert len(frequency) == F
        norm_factor = sound_velocity / (
            2 * frequency * shrink_factor * max_sensor_distance
        )
        norm_factor = np.nan_to_num(norm_factor)
        if norm_factor[-1] < 1:
            raise ValueError(
                'Distance between the sensors too high: {:.2} > {:.2}'.format(
                    max_sensor_distance, sound_velocity / (2 * frequency[-1])
                )
            )

        # Add empty dimensions at start and end
        norm_factor = norm_factor.reshape((signal.ndim - 2) * (1,) + (-1, 1,))

        signal = np.abs(signal) * np.exp(1j * np.angle(signal) * norm_factor)

    return signal


class Initializer:
    def __init__(self, Y_normalized, mixture_components, rng_state):
        self.Y_normalized = Y_normalized
        self.K = mixture_components
        self.rng_state = rng_state

    def iid_random_affiliations(self):
        affiliations = self.rng_state.dirichlet(
            self.K * [1 / self.K],
            size=(self.Y_normalized.shape[-2:])
        ).transpose((2, 0, 1))
        return dict(affiliations=affiliations)

    def frame_consistent_random_affiliations(self):
        affiliations = self.rng_state.dirichlet(
            self.K * [1 / self.K],
            size=(1, self.Y_normalized.shape[-1])
        ).transpose((2, 0, 1))
        affiliations = np.broadcast_to(
            affiliations,
            (self.K,) + Y.shape[-2:]
        )
        return dict(affiliations=affiliations)

    def iid_random_mode_vector(self):
        _, D, F, T = Y.shape
        W = np.zeros((self.K, F, D), dtype=Y.dtype)
        for f in range(F):
            choice = self.rng_state.choice(T, self.K, replace=False)
            W[:, f, :] = Y[0, :, f, choice]
        return dict(W=W)

    def frame_consistent_random_mode_vector(self):
        T = self.Y_normalized.shape[-1]
        choice = self.rng_state.choice(T, self.K, replace=False)
        W = reshape(self.Y_normalized[0, :, :, choice], 'dfk->kdf')
        return dict(W=W)


def initialize(initialization, Y_normalized, mixture_components, rng_state):
    _, D, F, T = Y_normalized.shape
    K = mixture_components

    if initialization is None:
        initialization = 'iid_random_affiliations'

    if isinstance(initialization, str):
        initializer = Initializer(Y_normalized, K, rng_state)
        initialization = getattr(initializer, initialization)()

    if 'affiliations' not in initialization:
        assert 'W' in initialization
        if 'kappa' not in initialization:
            initialization['kappa'] = 20 + np.zeros((K, F))
        if 'pi' not in initialization:
            initialization['pi'] = np.ones((K, F)) / K

    return initialization


class ComplexWatson:
    """
    >>> from os.TODO import bss
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> scales = [
    ...     np.arange(0, 0.01, 0.001),
    ...     np.arange(0, 20, 0.01),
    ...     np.arange(0, 100, 1)
    ... ]
    >>> functions = [
    ...     bss.ComplexWatson.log_norm_low_concentration,
    ...     bss.ComplexWatson.log_norm_medium_concentration,
    ...     bss.ComplexWatson.log_norm_high_concentration
    ... ]
    >>>
    >>> f, axis = plt.subplots(1, 3)
    >>> for ax, scale in zip(axis, scales):
    ...     result = [fn(scale, 6) for fn in functions]
    ...     [ax.plot(scale, np.log(r), '--') for r in result]
    ...     ax.legend(['low', 'middle', 'high'])
    >>> plt.show()
    """

    @staticmethod
    def pdf(x, loc, scale):
        """ Calculates pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        return np.exp(ComplexWatson.log_pdf(x, loc, scale))

    @staticmethod
    def log_pdf(x, loc, scale):
        """ Calculates logarithm of pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        # For now, we assume that the caller does proper expansion
        assert x.ndim == loc.ndim
        assert x.ndim - 1 == scale.ndim

        result = np.einsum('...d,...d', x, loc.conj())
        result = result.real ** 2 + result.imag ** 2
        result *= scale
        result -= ComplexWatson.log_norm(scale, x.shape[-1])
        return result

    @staticmethod
    def log_norm_low_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Good at very low concentrations but starts to drop of at 20.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        # Mardia1999Watson Equation (4), Taylor series
        b_range = range(dimension, dimension + 20 - 1 + 1)
        b_range = np.asarray(b_range)[None, :]

        return (
            np.log(2) + dimension * np.log(np.pi) -
            np.log(math.factorial(dimension - 1)) +
            np.log(1 + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_medium_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Almost complete range of interest and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.flatten()

        # Function is unstable at zero. Scale needs to be float for this to work
        scale[scale < 1e-2] = 1e-2

        r_range = range(dimension - 2 + 1)
        r = np.asarray(r_range)[None, :]

        # Mardia1999Watson Equation (3)
        temp = scale[:, None] ** r * np.exp(-scale[:, None]) / \
               np.asarray([math.factorial(_r) for _r in r_range])

        return (
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale +
            np.log(1. - np.sum(temp, -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_high_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        High concentration above 10 and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        return (
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale
        ).reshape(shape)

    log_norm = log_norm_medium_concentration


class HypergeometricRatioSolver:
    """ This is twice as slow as interpolation with Tran Vu's C-code, but works.

    >>> a = np.logspace(-3, 2.5, 100)
    >>> hypergeometric_ratio_inverse = HypergeometricRatioSolver()
    >>> hypergeometric_ratio_inverse([1/3, 0.5, 0.8, 1], 3)
    array([   0.        ,    2.68801168,    9.97621264,  100.        ])
    >>> hypergeometric_ratio_inverse(a, 3)
    """

    # TODO: Is it even necessary to cache this?
    # TODO: Possibly reduce number of markers, if speedup is necessary at all.
    # TODO: Improve handling for very high and very low values.

    def __init__(self, max_concentration=100, markers=100):
        x = np.logspace(-3, np.log10(max_concentration), markers)
        self.x = x
        self.max_concentration = max_concentration

    @functools.lru_cache(maxsize=3)
    def _get_spline(self, D):
        y = hyp1f1(2, D + 1, self.x) / (D * hyp1f1(1, D, self.x))
        return interp1d(
            y, self.x, kind='quadratic',
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, self.max_concentration)
        )

    def __call__(self, a, D):
        return self._get_spline(D)(a)


def em(
        Y, mixture_components=3, iterations=100,
        initialization=None, alignment=True, rng_state=np.random,
        spatially_white_noise_assumption=False, max_concentration=100
):
    """

    Args:
        Y:
        mixture_components:
        iterations:
        initialization: Can be a dict containing either affiliations or
            a subset of {W, kappa, pi}.
        alignment:
        rng_state:
        spatially_white_noise_assumption:
        max_concentration:

    Returns:

    """
    Y_normalized = normalize_observation(Y, frequency_norm=False)
    Y_normalized_for_psd = np.copy(Y_normalized[0], 'C')
    Y_normalized_for_pdf = np.copy(Y_normalized.transpose(0, 2, 3, 1), 'C')

    hypergeometric_ratio_inverse = HypergeometricRatioSolver(
        max_concentration=max_concentration
    )

    initialization = initialize(
        initialization, Y_normalized, mixture_components, rng_state
    )

    for i in range(iterations):
        if i == 0 and 'affiliations' in initialization:
            affiliations = initialization['affiliations']
        else:
            if i == 0 and 'W' in initialization:
                W = initialization['affiliations']
                kappa = initialization['kappa']
                pi = initialization['pi']
            affiliations = pi[..., None] * ComplexWatson.pdf(
                Y_normalized_for_pdf,
                np.copy(W[:, :, None, :], 'C'),
                kappa[:, :, None]
            )
            affiliations /= np.sum(affiliations, axis=0, keepdims=True)

        pi = affiliations.mean(axis=-1)
        Phi = get_power_spectral_density_matrix(
            Y_normalized_for_psd,
            np.copy(affiliations, 'C'),
            sensor_dim=0, source_dim=0,
            time_dim=-1
        )
        W, eigenvalues = get_pca(Phi)
        kappa = hypergeometric_ratio_inverse(eigenvalues, W.shape[-1])

        if spatially_white_noise_assumption:
            kappa[-1, ...] = 0

    if alignment:
        mapping = bss.frequency_permutation_alignment(affiliations)
        bss.apply_alignment_inplace(affiliations, pi, W, kappa, mapping=mapping)

    return dict(
        affiliations=affiliations,
        kappa=kappa,
        W=W,
        pi=pi
    )
