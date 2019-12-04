import numpy as np


def normalize_mean_variance(data, axis=0, eps=1e-6):
    """ Normalize features.

    :param data: Any feature
    :param axis: Time dimensions, default is 0
    :return: Normalized observation
    """
    return ((data - np.mean(data, axis=axis, keepdims=True)) /
            (np.std(data, axis=axis, keepdims=True) + eps))
