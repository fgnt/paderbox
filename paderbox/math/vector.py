import numpy as np

def cos_similarity(A, B):
    """
    returns cosine similarity between array A and B
    Args:
        A: array with shape (...,d)
        B: array with shape (...,d)

    Returns:
        cosine similarity with shape (...)

    """
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
    return 0.5 * (1 - sum(a * b) / np.sqrt(sum(a ** 2) * sum(b ** 2)))


def normalize_vector_to_unit_length(data):
    """
    Normalized each vector to unit length. This is useful, if all other
    normalization techniques are not reliable.

    :param data: Assumes an input with shape (..., vector)
    :return: The input with the last dimension normalized
    """
    normalization = np.sqrt(np.abs(vector_H_vector(data, data)))
    return data / np.expand_dims(normalization, axis=-1)


def vector_H_vector(x, y):
    """ Inner product of last array dimensions.

    :param x: LHS. Same shape as y
    :param y: RHS. Same shape as x
    :return:
    """
    return np.einsum('...a,...a->...', x.conj(), y)


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def cart2sph(x, y, z):
    """transforms cartesian to spherical coordinates"""
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# http://stackoverflow.com/questions/30084174/efficient-matlab-cart2sph-and-sph2cart-functions-in-python
def sph2cart(azimuth, elevation, r):
    """transforms spherical to cartesian coordinates"""
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z