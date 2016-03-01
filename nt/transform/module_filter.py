"""
Provides general filters, for example preemphasis filter.
"""
from scipy.signal import lfilter, medfilt


def preemphasis(time_signal, p=0.95):
    """
    Pre-emphasis filter.
    """
    return lfilter([1., -p], [1], time_signal)


def inverse_preemphasis(time_signal, p=0.95):
    """
    Pre-emphasis filter.
    """
    return lfilter([1], [1., -p], time_signal)


def offset_compensation(time_signal):
    """
    Offset compensation filter.
    """
    return lfilter([1., -1], [1., -0.999], time_signal)


def median(input, window_size=3):
    """
    Median Filter
    :param input:array of values to be filtered
    :param window_size: kernel size for the filter
    :return: filtered output array of same length as input
    """
    return medfilt(input, window_size)
