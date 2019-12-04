"""
This module deals with all sorts of acoustic features and transforms.
"""
from .module_stft import (
    stft,
    istft,
    STFT,
    spectrogram,
    stft_to_spectrogram,
    spectrogram_to_energy_per_frame,
    get_stft_center_frequencies,
)

import numpy as np
from paderbox.transform.module_resample import resample_sox


def normalize_mean_variance(data, axis=0, eps=1e-6):
    """ Normalize features.
from .module_filter import (
    preemphasis,
    inverse_preemphasis,
    offset_compensation,
    preemphasis_with_offset_compensation,
)

    :param data: Any feature
    :param axis: Time dimensions, default is 0
    :return: Normalized observation
    """
    return (data - np.mean(data, axis=axis, keepdims=True)) /\
        (np.std(data, axis=axis, keepdims=True) + eps)
from .module_fbank import fbank, logfbank
from .module_mfcc import mfcc, mfcc_velocity_acceleration
from .module_normalize import normalize_mean_variance
from .module_ssc import ssc
from .module_bark_fbank import bark_fbank
from .module_rastaplp import rasta_plp
from .module_ams import ams
from .module_resample import resample_sox
