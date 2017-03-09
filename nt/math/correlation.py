import numpy as np

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
            normalization = np.sum(mask, axis=-1, keepdims=True)
            psd /= normalization

    return psd