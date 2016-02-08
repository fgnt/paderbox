import os.path

import numpy as np

from nt.utils.matlab import Mlab
from nt.utils.numpy_utils import segment_axis

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
    if not mlab.process.started:
        mlab.process.start()
    else:
        mlab.run_code('clear all;')

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
