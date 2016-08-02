import numpy as np
from nt.transform import get_stft_center_frequencies
import math


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
        signal, phase_norm=True, unit_norm=True, frequency_norm=True,
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

    if phase_norm:
        signal *= np.exp(-1j * np.angle(signal[..., 0, :, :]))

    if unit_norm:
        signal /= np.linalg.norm(signal, axis=-3, keepdims=True)

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


class ComplexWatson:
    """
    >>> from nt.speech_enhancement import bss
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
        temp = scale[:, None]**r * np.exp(-scale[:, None]) / \
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
