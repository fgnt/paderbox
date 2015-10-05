
import numpy as np

def normalize_vector_to_unit_length(vector):
    """
    Normalized each vector to unit length. This is useful, if all other
    normalization techniques are not reliable.

    :param vector: Assumes a beamforming vector with shape (bins, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    normalization = np.sqrt(np.abs(vector_H_vector(vector, vector)))
    return vector / normalization[:, np.newaxis]


def vector_H_vector(x, y):
    return np.einsum('...a,...a->...', x.conj(), y)
