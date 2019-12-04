"""
Provides general filters, for example preemphasis filter.
"""
from scipy.signal import lfilter, medfilt


def preemphasis(time_signal, p=0.95):
    """Default Pre-emphasis filter.

    Performs a causal IIR filter with the transfer function
        H(z) = 1 - p*z**(-1)
    to lower low-frequency and increase high-frequency components.

    :param time signal: The input signal to be filtered.
    :param p: preemphasis coefficient
    :return: The filtered input signal
    """
    return lfilter([1., -p], [1], time_signal)


def inverse_preemphasis(time_signal, p=0.95):
    """Inverse Pre-emphasis filter.

    Removes the effect of preemphasis.

    :param time signal: The input signal to be filtered.
    :param p: preemphasis coefficient
    :return: The filtered input signal
    """
    return lfilter([1], [1., -p], time_signal)


def offset_compensation(time_signal):
    """ Offset compensation filter.
    """
    return lfilter([1., -1], [1., -0.999], time_signal)


def preemphasis_with_offset_compensation(time_signal, p=0.95):
    """Combined filter to add pre-emphasis and compensate the offset.

    This approach offers increased numerical accuracy.

    :param time signal: The input signal to be filtered.
    :param p: preemphasis coefficient
    :return: The filtered input signal
    """
    return lfilter([1, -(1+p), p], [1, -0.999], time_signal)


def median(input_signal, window_size=3):
    """ Median Filter

    :param input_signal: array of values to be filtered
    :param window_size: kernel size for the filter
    :return: filtered output signal of same length as input_signal
    """
    return medfilt(input_signal, window_size)
