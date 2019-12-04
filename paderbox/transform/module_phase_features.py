from copy import deepcopy

import numpy as np
import paderbox.math.directional as directional


def transform_to_baseband(X, size, shift):
    """Assumes linear frequency dependency.

    Then phase is more consistent over frequencies.

    Args:
        X:
        size:
        shift:

    Returns:

    """
    X = deepcopy(X)
    T, _, F = X.shape
    for t in range(T):
        for f in range(F):
            X[t, :, f] *= np.exp(-2j * np.pi * t * f * shift / size)
    return X


def get_phase_features(X, size, shift):
    """Experimental phase features for SPP estimation.

    These experimental features were originally used because the phase
    differences, at least when small STFT shifts are used, correspond better
    to SPP than the phase itself.

    Args:
        X:
        size:
        shift:

    Returns:

    """
    X_base = transform_to_baseband(X, size, shift)
    phase = np.angle(X_base)
    delta = np.zeros_like(phase)
    delta[1:, ...] = directional.minus(phase[1:, ...], phase[:-1, ...])
    delta_delta = np.zeros_like(phase)
    delta_delta[1:, ...] = delta[1:, ...] - delta[:-1, ...]
    return phase, delta, delta_delta
