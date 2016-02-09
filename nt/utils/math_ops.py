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
    An example is F times M times T.

    :param x: Signal with dimensions ... times M times T.
    :param mask: Mask with dimensions ... times T.
    :return: Covariance matrices with dimensions ... times M times M.
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


def scaled_full_correlation_matrix(X, iterations=4, trace_one_constraint=True):
    """ Scaled full correlation matrix.

    See the paper "Generalization of Multi-Channel Linear Prediction
    Methods for Blind MIMO Impulse Response Shortening" for reference.

    You can plot the time dependent power to validate this function.

    :param X: Assumes shape F, M, T.
    :param iterations: Number of iterations between time dependent
        scaling factor (power) and covariance estimation.
    :param trace_one_constraint: This constraint is not part of the original
        paper. It is not necessary for the result but removes the scaling
        ambiguity.
    :return: Covariance matrix and time dependent power for each frequency.
    """
    F, M, T = X.shape

    def _normalize(cm):
        if trace_one_constraint:
            trace = np.einsum('...mm', cm)
            cm /= trace[:, None, None] / M
            cm += 1e-6 * np.eye(M)[None, :, :]
        return cm

    covariance_matrix = covariance(X)
    covariance_matrix = _normalize(covariance_matrix)

    for i in range(iterations):
        inverse = np.linalg.inv(covariance_matrix)

        power = np.abs(np.einsum(
            '...mt,...mn,...nt->...t',
            X.conj(),
            inverse,
            X
        )) / M

        covariance_matrix = covariance(X, mask=1/power)
        covariance_matrix = _normalize(covariance_matrix)
    return covariance_matrix, power


def cos_similarity(A, B):
    similarity = np.abs(np.einsum('...d,...d', A, B.conj()))
    similarity /= np.sqrt(np.abs(np.einsum('...d,...d', A, A.conj())))
    similarity /= np.sqrt(np.abs(np.einsum('...d,...d', B, B.conj())))
    return similarity
