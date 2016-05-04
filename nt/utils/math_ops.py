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


def covariance(x, mask=None, normalize=True):
    """ Calculate the covariance of a zero mean signal.

    The leading dimensions are independent and can be arbitrary.
    An example is F times M times T.

    :param x: Signal with dimensions ... times M times T.
    :param mask: Mask with dimensions ... times T.
    :return: Covariance matrices with dimensions ... times M times M.
    """

    if mask is None:
        psd = np.einsum('...dt,...et->...de', x, x.conj())
        if normalize:
            psd /= x.shape[-1]
    else:
        assert x.ndim == mask.ndim + 1, 'Wrong total number of dimensions.'
        assert x.shape[-1] == mask.shape[-1], 'Check dimension for summation.'
        mask = np.expand_dims(mask, -2)
        psd = np.einsum('...dt,...et->...de', mask * x, x.conj())
        if normalize:
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


def cos_distance(a, b):
    """
    cosine distance between vector a and b: 1/2*(1-a*b/|a|*|b|)

    :param a: vector a (1xN or Nx1 numpy array)
    :param b: vector b (1xN or Nx1 numpy array)
    :return: distance (scalar)
    """
    return 0.5*(1 - sum(a*b)/np.sqrt(sum(a**2)*sum(b**2)))


def _calculate_block_boundaries(T, block_size, first_block_size):
    block_boundaries = []
    block_start = 0
    block_end = np.minimum(first_block_size, T)
    while True:
        block_boundaries.append((block_start, block_end))
        if block_end == T:
            break
        block_start = block_end
        block_end = np.minimum(block_start + block_size, T)
    return block_boundaries


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def cart2sph(x, y, z):
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
