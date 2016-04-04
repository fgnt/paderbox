import numpy as np


def select_features(signal, bins=(5, 190)):
    """ Helps to truncate features.

    :param signal: Input features with format T, D, F
    :param bins: Tuple of start and end of desired features
    :return: Truncated features
    """
    if signal.ndim == 2:
        return signal[:, bins[0]:bins[1]]
    elif signal.ndim == 3:
        return signal[:, :, bins[0]:bins[1]]
    else:
        raise NotImplementedError


def expand_features(signal, bins=(5, 190), total=513):
    """ Helps to expand previously truncated features.

    :param signal: Truncated features with format T, D, F'
    :param bins: Tuple of start and end truncated features
    :param total: Total number of features
    :return: Features with original shape filled with zeros.
    """
    if signal.ndim == 2:
        T, _ = signal.shape
        temp = np.zeros((T, total), dtype=signal.dtype)
        temp[:, bins[0]:bins[1]] = signal
    elif signal.ndim == 3:
        T, D, _ = signal.shape
        temp = np.zeros((T, D, total), dtype=signal.dtype)
        temp[:, :, bins[0]:bins[1]] = signal
    else:
        raise NotImplementedError
    return temp
