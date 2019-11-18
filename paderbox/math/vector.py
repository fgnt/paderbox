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
