import numpy as np

def normalize_vector_to_unit_length(vector):
    """
    Normalized each vector to unit length. This is useful, if all other
    normalization techniques are not reliable.

    :param vector: Assumes a beamforming vector with shape (bins, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    normalization = np.sqrt(np.abs(vector_H_vector(vector, vector)))
    return vector / np.expand_dims(normalization, axis=-1)


def vector_H_vector(x, y):
    return np.einsum('...a,...a->...', x.conj(), y)


def softmax(x, feature_axis=-1):
    """ Calculates the softmax activation

    :param x: Input signal
    :param feature_axis: Dimension holding the features to apply softmax on
    :return: Softmax features
    """
    net_out_e = x - x.max(axis=feature_axis, keepdims=True)
    np.exp(net_out_e, out=net_out_e)
    net_out_e /= net_out_e.sum(axis=feature_axis, keepdims=True)
    return net_out_e


def covariance(x, mask=None):
    """ Calculate the covariance of a zero mean signal.

    The leading dimensions are independent and can be arbitrary.

    :param x: Signal with dimensions ... times F times M times T.
    :param mask: Mask with dimensions ... times F times T.
    :return: Covariance matrices with dimensions ... times F times M times M.
    """

    if mask is None:
        psd = np.einsum('...dt,...et->...de', x, x.conj())
        psd /= x.shape[-1]
    else:
        assert x.ndim == mask.ndim + 1, 'Wrong total number of dimensions.'
        assert x.shape[-1] == mask.shape[-1], 'Check dimension for summation.'
        mask = np.expand_dims(mask, -2)
        psd = np.einsum('...dt,...et->...de', mask * x, x.conj())
        normalization = np.maximum(np.sum(mask, axis=-1, keepdims=True), 1e-10)
        psd /= normalization

    return psd
