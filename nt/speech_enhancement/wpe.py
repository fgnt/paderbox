import os.path

import numpy as np
from nt.math.correlation import covariance
from pathlib import Path

from nt.utils import deprecated

try:
    from nt.utils.matlab import Mlab

    matlab_available = True
except ImportError:
    matlab_available = False
from nt.utils.numpy_utils import segment_axis

if matlab_available:
    mlab = Mlab()


def dereverb(settings_file_path, x, stop_mlab=True):
    """
    This method wraps the matlab WPE-dereverbing-method. Give it the path to
    the settings.m and the wpe.p file and your reverbed signals as numpy matrix.
    Return value will be the dereverbed signals as numpy matrix.

    .. note:: The overall settings for this method are determined in the
        settings.m file. The wpe.p needs that settings.m file as input argument
        in order to work properly. Make sure that you read your audio signals
        accordingly.

    .. warning:: The settings file name MUST be 'wpe_settings'!

    :param settings_file_path: Path to wpe_settings.m and wpe.p
    :param x: NxC Numpy matrix of read audio signals. N denotes the signals'
        number of frames and C stands for the number of channels you provide
        for that signal
    :param stop_mlab: Whether matlab connection should be closed after execution
    :return: NxC Numpy matrix of dereverbed audio signals. N and C as above.
    """
    if not matlab_available:
        raise EnvironmentError('Matlab not available')
    if not mlab.process.started:
        mlab.process.start()
    else:
        mlab.run_code('clear all;')

    if isinstance(settings_file_path, Path):
        settings_file_path = str(settings_file_path)

    settings = os.path.join(settings_file_path, "wpe_settings.m")

    # Check number of channels and set settings.m accordingly
    c = x.shape[1]
    modify_settings = False
    lines = []
    with open(settings) as infile:
        for line in infile:
            if 'num_mic = ' in line:
                if not str(c) in line:
                    line = 'num_mic = ' + str(c) + ";\n"
                    modify_settings = True
                else:
                    break  # ignore variable lines
            lines.append(line)
    if modify_settings:
        with open(settings, 'w') as outfile:
            for line in lines:
                outfile.write(line)

    # Process each utterance
    mlab.set_variable("x", x)
    mlab.set_variable("settings", settings)
    assert np.allclose(mlab.get_variable("x"), x)
    assert mlab.get_variable("settings") == settings
    mlab.run_code("addpath('" + settings_file_path + "');")

    # start wpe
    print("Dereverbing ...")
    mlab.run_code("y = wpe(x, settings);")
    # write dereverbed audio signals
    y = mlab.get_variable("y")

    if mlab.process.started and stop_mlab:
        mlab.process.stop()
    return y


@deprecated('use nara_wpe')
def wpe(Y, epsilon=1e-6, order=15, delay=1, iterations=10):
    """

    :param Y: Stft signal (TxF)
    :param epsilon:
    :param order: Linear prediction order
    :param delay: Prediction delay
    :param iterations: Number of iterations
    :return: Dereverberated Stft signal
    """
    T, F = Y.shape
    dtype = Y.dtype
    power_spectrum = np.maximum(np.abs(Y * Y.conj()), epsilon)
    dereverberated = np.zeros_like(Y)

    for iteration in range(iterations):
        regression_coefficient = np.zeros((F, order), dtype=dtype)
        Y_norm = Y / np.sqrt(power_spectrum)
        Y_windowed = segment_axis(
            Y,
            order,
            order - 1,
            axis=0).T[..., :-delay - 1]
        Y_windowed_norm = segment_axis(Y_norm,
                                       order, order - 1,
                                       axis=0, ).T[..., :-delay - 1]
        correlation_matrix = np.einsum('...dt,...et->...de', Y_windowed_norm,
                                       Y_windowed_norm.conj())
        cross_correlation_vector = np.sum(
            Y_windowed_norm * Y_norm[order + delay:, None, :].T.conj(), axis=-1)
        for f in range(F):
            regression_coefficient[f, :] = np.linalg.solve(
                correlation_matrix[f, :, :], cross_correlation_vector[f, :])
        regression_signal = np.einsum('ab,abc->ac',
                                      regression_coefficient.conj(),
                                      Y_windowed).T
        dereverberated[order + delay:, :] = \
            Y[order + delay:, :] - regression_signal
        power_spectrum = np.maximum(
            np.abs(dereverberated * dereverberated.conj()), epsilon)

    return dereverberated


@deprecated('use nara_wpe')
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


@deprecated('use nara_wpe')
def _dereverberate(y, G_hat, K, Delta):
    L, N, T = y.shape
    dtype = y.dtype
    x_hat = np.copy(y)
    for l in range(L):
        for t in range(Delta + K, T):  # Some restrictions
            for tau in range(Delta, Delta + K):
                x_hat[l, :, t] -= G_hat[l, tau - Delta, :, :].conj().T.dot(
                    y[l, :, t - tau])
    return x_hat


@deprecated('use nara_wpe')
def _dereverberate_vectorized(y, G_hat, K, Delta):
    x_hat = np.copy(y)
    for tau in range(Delta, Delta + K):
        x_hat[:, :, K + Delta:] -= np.einsum('abc,abe->ace',
                                             G_hat[:, tau - Delta, :, :].conj(),
                                             y[..., K + Delta - tau:-tau])
    return x_hat


@deprecated('use nara_wpe')
def _get_spatial_correlation_matrix_inverse(y):
    L, N, T = y.shape
    correlation_matrix, power = scaled_full_correlation_matrix(y)
    # Lambda_hat = correlation_matrix[:, :, :, None] * power[:, None, None, :]
    # inverse = np.zeros_like(Lambda_hat)
    # for l in range(L):
    #     for t in range(T):
    #         inverse[l, :, :, t] = np.linalg.inv(Lambda_hat[l, :, :, t])
    inverse = np.zeros_like(correlation_matrix)
    for l in range(L):
        inverse[l, :, :] = np.linalg.inv(correlation_matrix[l, :, :])

    inverse = inverse[:, :, :, None] / power[:, None, None, :]
    return inverse


@deprecated('use nara_wpe')
def _get_crazy_matrix(Y, K, Delta):
    # A view may possibly be enough as well.
    L, N, T = Y.shape
    dtype = Y.dtype
    psi_bar = np.zeros((L, N * N * K, N, T - Delta - K + 1), dtype=dtype)
    for n0 in range(N):
        for n1 in range(N):
            for tau in range(Delta, Delta + K):
                for t in range(T):
                    psi_bar[
                    :, N * N * (tau - Delta) + N * n0 + n1, n0,
                    t - Delta - K + 1
                    ] = Y[:, n1, t - tau]
    return psi_bar


@deprecated('use nara_wpe')
def _get_Phi_YY(Y, l, t_1, t_2):
    phi_yy = np.outer(Y[l, :, t_1], Y[l, :, t_2].conj())
    assert np.all(np.abs(phi_yy) > 0)
    return phi_yy


@deprecated('use nara_wpe')
def _get_T_segmented(Y, l, t, K):
    assert Y.ndim == 3
    N = Y.shape[1]
    T = np.zeros((N * N * K, N * N * K), dtype=Y.dtype)
    for index_1, t_1 in enumerate(range(t, t - K, -1)):
        for index_2, t_2 in enumerate(range(t, t - K, -1)):
            T[index_1 * N * N:(index_1 + 1) * N * N,
            index_2 * N * N:(index_2 + 1) * N * N] = np.tile(
                _get_Phi_YY(Y, l, t_1, t_2), (N, N))
    np.testing.assert_almost_equal(T, T.T.conj())
    return T


@deprecated('use nara_wpe')
def _get_T_segmented_prediction(Y, l, t_m_delta, t, K):
    assert Y.ndim == 3
    N = Y.shape[1]
    T = np.zeros((N * N * K, N * N), dtype=Y.dtype)
    for index_1, t_1 in enumerate(range(t_m_delta, t_m_delta - K, -1)):
        T[index_1 * N * N:(index_1 + 1) * N * N, :] = np.tile(
            _get_Phi_YY(Y, l, t_1, t), (N, N))
    assert np.all(np.abs(T) > 0)
    return T


@deprecated('use nara_wpe')
def _get_global_T_segmented(Y, K, Delta):
    L, N, T = Y.shape
    global_T = np.zeros((L, N * N * K, N * N * K, T - K - Delta), dtype=Y.dtype)
    for l in range(L):
        for t in range(K + Delta, T):
            global_T[l, :, :, t - K - Delta] = _get_T_segmented(Y, l, t, K)
    assert np.all(np.abs(global_T) > 0)
    np.testing.assert_almost_equal(global_T,
                                   global_T.transpose(0, 2, 1, 3).conj())
    return global_T


@deprecated('use nara_wpe')
def _get_global_T_segmented_prediction(Y, K, Delta):
    L, N, T = Y.shape
    global_T = np.zeros((L, N * N * K, N * N, T - K - Delta), dtype=Y.dtype)
    for l in range(L):
        for t in range(K + Delta, T):
            global_T[l, :, :, t - K - Delta] = _get_T_segmented_prediction(Y, l,
                                                                           t - Delta,
                                                                           t, K)
    assert np.all(np.abs(global_T) > 0)
    return global_T


@deprecated('use nara_wpe')
def _y_tilde(Y, l, t):
    L, N, T = Y.shape
    y_tilde = np.zeros((N*N, N), dtype=Y.dtype)
    for n in range(N):
        y_tilde[n*N:(n+1)*N, n] = Y[l, :, t]
    return y_tilde


@deprecated('use nara_wpe')
def _psi(Y, l, t, K):
    assert t>=K-1
    L, N, T = Y.shape
    psi = np.zeros((N*N*K, N), dtype=Y.dtype)
    for k in range(K):
        psi[k*N*N:(k+1)*N*N] = _y_tilde(Y, l, t-k)
    return psi


@deprecated('use nara_wpe')
def multichannel_wpe(Y, K, Delta, iterations=4):
    # K: regression_order (possibly frequency dependent)
    # Delta: prediction_delay
    # L: frequency bins
    # N: sensors
    # T: time frames
    L, N, T = Y.shape
    dtype = Y.dtype

    # Step 1
    G_hat = np.zeros((L, K, N, N), dtype=dtype)

    for _ in range(iterations):
        # Step 2
        x_hat = _dereverberate(Y, G_hat, K, Delta)
        assert x_hat.shape == (L, N, T)

        # Step 3
        # Maybe better on a subpart, due to fade in
        Lambda_hat_inverse = _get_spatial_correlation_matrix_inverse(x_hat)

        # Step 4
        L, N, T = Y.shape
        R_hat = np.zeros((L, N*N*K, N*N*K), dtype=Y.dtype)
        r_hat = np.zeros((L, N*N*K), dtype=Y.dtype)
        for l in range(L):
            for t in range(K+Delta, T):
                psi_bar = _psi(Y, l, t-Delta, K).conj()
                R_hat[l, :, :] = psi_bar.dot(Lambda_hat_inverse[l, :, :, t]).dot(psi_bar.T.conj())
                r_hat[l, :] = psi_bar.dot(Lambda_hat_inverse[l, :, :, t]).dot(Y[l, :, t])

        # Step 5
        # the easiness of the reshape depends on the definition of psi_bar
        g_hat = np.zeros((L, N * N * K), dtype=dtype)
        for l in range(L):
            # g_hat[l, :] = np.linalg.inv(R_hat[l, :, :]).dot(r_hat[l, :])
            g_hat[l, :] = np.linalg.solve(R_hat[l, :, :], r_hat[l, :])
        assert g_hat.shape == (L, N * N * K)
        G_hat = g_hat.reshape(L, N, N, K).transpose((0, 3, 1, 2))
        assert G_hat.shape == (L, K, N, N)

    return x_hat
