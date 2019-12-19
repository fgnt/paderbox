import numpy as np
from paderbox.transform.module_stft import stft
from paderbox.transform.module_fbank import logfbank
from paderbox.array import segment_axis
import scipy.signal
from scipy.fftpack import dct


def mfcc(time_signal, sample_rate=16000,
         window_length=400, stft_shift=160, numcep=13,
         number_of_filters=26, stft_size=512,
         lowest_frequency=0, highest_frequency=None,
         preemphasis_factor=0.97, ceplifter=22,
         window=scipy.signal.hamming):
    """
    Compute MFCC features from an audio signal.

    Source: https://github.com/jameslyons/python_speech_features

    Illustrations: http://ntjenkins.upb.de/view/PythonToolbox/job/python_toolbox_notebooks/HTML_Report/build/toolbox_examples/transform/06%20-%20Additional%20features.html

    :param time_signal: the audio signal from which to compute features.
        Should be an channels x samples array.
    :param sample_rate: the sample rate of the signal we are working with.
        Default is 16000.
    :param window_length: the length of the analysis window. In samples.
        Default is 400 (25 milliseconds @ 16kHz).
    :param stft_shift: the step between successive windows. In samples.
        Default is 160 (10 milliseconds @ 16kHz).
    :param numcep: the number of cepstrum to return, Default is 13.
    :param number_of_filters: number of filters in the filterbank,
        Default is 26.
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of mel filters. In Hz,
        Default is 0.
    :param highest_frequency: highest band edge of mel filters. In Hz,
        Default is samplerate/2.
    :param preemphasis_factor: apply preemphasis filter with preemphasis_factor
        as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: the liftering coefficient to use.
        ceplifter <= 0 disables lifter.
        Default is 22.
    :param window: the window function to use for fbank features. Default is
        hamming window.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features.
        Each row holds 1 feature vector.
    """
    feat = logfbank(
        time_signal, sample_rate, window_length, stft_shift,
        number_of_filters, stft_size, lowest_frequency,
        highest_frequency, preemphasis_factor, window)
    feat = dct(feat, type=2, axis=-1, norm='ortho')[..., :numcep]
    feat = _lifter(feat, ceplifter)

    return feat


def _lifter(cepstra, L=22):
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coefficients.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in
        size.
    :param L: the liftering coefficient to use. Default is 22.
        L <= 0 disables lifter.

    Source: https://github.com/jameslyons/python_speech_features
    Tutorial MFCC: http://practicalcryptography.com/miscellaneous/
        machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)[-2:]
        n = np.arange(ncoeff)
        lift = 1+ (L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def mfcc_velocity_acceleration(time_signal, *args, **kwargs):
    """ Calculate MFCC velocity and acceleration.

    The deltas are calculated just as in Kaldi:
    https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-functions.cc#L235

    The ETSI standard frontend goes explains the same on page 39 section 9.2:
    http://www.etsi.org/deliver/etsi_es%5C202000_202099%5C202050%5C01.01.05_60%5Ces_202050v010105p.pdf

    :param time_signal: Time signal
    :param args: All parameters for MFCCs
    :param kwargs: All parameters for MFCCs
    :return: Stacked features
    """
    mfcc_signal = mfcc(time_signal, *args, **kwargs)
    delta_mfcc_signal = delta(mfcc_signal, order=1)
    delta_delta_mfcc_signal = delta(mfcc_signal, order=2)
    return np.concatenate(
        (mfcc_signal, delta_mfcc_signal, delta_delta_mfcc_signal),
        axis=1
    )


def delta(data, width=9, order=1, axis=-1, trim=True):
    r'''Compute delta features: local estimate of the derivative
    of the input data along the selected axis.


    Parameters
    ----------
    data      : np.ndarray
        the input data matrix (eg, spectrogram)

    width     : int >= 3, odd [scalar]
        Number of frames over which to compute the delta feature

    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.

    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).

    trim      : bool
        set to `True` to trim the output matrix to the original size.

    Returns
    -------
    delta_data   : np.ndarray [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.

    '''

    data = np.atleast_1d(data)

    if width < 3 or np.mod(width, 2) != 1:
        raise ValueError('width must be an odd integer >= 3')

    if order <= 0 or not isinstance(order, int):
        raise ValueError('order must be a positive integer')

    half_length = 1 + int(width // 2)
    window = np.arange(half_length - 1., -half_length, -1.)

    # Normalize the window so we're scale-invariant
    window /= np.sum(np.abs(window)**2)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    width = int(width)
    padding[axis] = (width, width)
    delta_x = np.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)

    # Cut back to the original shape of the input data
    if trim:
        idx = [slice(None)] * delta_x.ndim
        idx[axis] = slice(- half_length - data.shape[axis], - half_length)
        delta_x = delta_x[idx]

    return delta_x


def modmfcc(
        time_signal, sample_rate=16000,
        stft_win_len=400, stft_shift=160, numcep=30,
        number_of_filters=40, stft_size=512,
        lowest_frequency=0, highest_frequency=None,
        preemphasis_factor=0.97, ceplifter=22,
        stft_window=scipy.signal.hamming,
        mod_length=16, mod_shift=8, mod_window=scipy.signal.hamming,
        avg_length=1, avg_shift=1
):
    """
    Compute Mod-MFCC features from an audio signal.

    :param time_signal: the audio signal from which to compute features.
        Should be an channels x samples array.
    :param sample_rate: the sample rate of the signal we are working with.
        Default is 16000.
    :param stft_win_len: the length of the analysis window. In samples.
        Default is 400 (25 milliseconds @ 16kHz).
    :param stft_shift: the step between successive windows. In samples.
        Default is 160 (10 milliseconds @ 16kHz).
    :param numcep: the number of cepstrum to return, Default is 20.
    :param number_of_filters: number of filters in the filterbank,
        Default is 40.
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of mel filters. In Hz,
        Default is 0.
    :param highest_frequency: highest band edge of mel filters. In Hz,
        Default is samplerate/2.
    :param preemphasis_factor: apply preemphasis filter with preemphasis_factor
        as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: the liftering coefficient to use.
        ceplifter <= 0 disables lifter.
        Default is 22.
    :param stft_window: the window function to use for fbank features. Default is
        hamming window.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features.
        Each row holds 1 feature vector.
    """
    x = mfcc(
        time_signal, sample_rate=sample_rate, window_length=stft_win_len,
        window=stft_window, stft_shift=stft_shift, stft_size=stft_size,
        number_of_filters=number_of_filters, lowest_frequency=lowest_frequency,
        highest_frequency=highest_frequency,
        preemphasis_factor=preemphasis_factor,
        ceplifter=ceplifter, numcep=numcep)

    x = np.abs(stft(
        x,
        size=mod_length, shift=mod_shift, window=mod_window,
        axis=-2, fading=False
    ))
    assert avg_length >= avg_shift
    if avg_length > 1:
        x = segment_axis(
            x, length=avg_length, shift=avg_shift, end='pad',
            axis=-3)
        x = np.mean(x, axis=-3)
    return x
