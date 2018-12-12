from copy import deepcopy

import numpy as np
import nt.math.directional as directional


def transform_to_baseband(X, size, shift):
    X = deepcopy(X)
    T, _, F = X.shape
    for t in range(T):
        for f in range(F):
            X[t, :, f] *= np.exp(-2j * np.pi * t * f * shift / size)
    return X


def get_phase_features(X, size, shift):
    X_base = transform_to_baseband(X, size, shift)
    phase = np.angle(X_base)
    delta = np.zeros_like(phase)
    delta[1:, ...] = directional.minus(phase[1:, ...], phase[:-1, ...])
    delta_delta = np.zeros_like(phase)
    delta_delta[1:, ...] = delta[1:, ...] - delta[:-1, ...]
    return phase, delta, delta_delta
