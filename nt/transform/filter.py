"""
Provides general filters, for example preemphasis filter.
"""
from scipy.signal import lfilter

def preemphasis(input, p):
    """
    Pre-emphasis filter.
    """
    return lfilter([1., -p], 1, input)

def offcomp(input):
    """
    Offset compensation filter.
    """
    return lfilter([1., -1], [1., -0.999], input)
