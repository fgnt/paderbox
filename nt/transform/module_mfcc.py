import numpy
import numpy as np
from nt.transform.module_fbank import fbank
import scipy.signal
from scipy.fftpack import dct
from librosa.feature import delta


def mfcc(time_signal, sample_rate=16000,
         window_length=400, stft_shift=160, numcep=13,
         number_of_filters=26, stft_size=512,
         lowest_frequency=0, highest_frequency=None,
         preemphasis_factor=0.97, ceplifter=22,
         window=scipy.signal.hamming):
    """
    Compute MFCC features from an audio signal.

    Source: https://github.com/jameslyons/python_speech_features

    Illustrations: http://ntjenkins.upb.de/view/PythonToolbox/job/
        python_toolbox_notebooks/HTML_Report/toolbox_examples/
        transform/06%20-%20Additional%20features.html

    :param time_signal: the audio signal from which to compute features.
        Should be an N*1 array
    :param sample_rate: the samplerate of the signal we are working with.
    :param window_length: the length of the analysis window in seconds.
        Default is 0.025s (25 milliseconds)
    :param stft_shift: the step between successive windows in seconds.
        Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param number_of_filters: the number of filters in the filterbank,
        default 26.
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of mel filters. In Hz,
        default is 0.
    :param highest_frequency: highest band edge of mel filters. In Hz,
        default is samplerate/2
    :param preemphasis: apply preemphasis filter with preemph as coefficient.
        0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features.
        Each row holds 1 feature vector.
    """
    feat = fbank(time_signal, sample_rate, window_length, stft_shift,
                       number_of_filters, stft_size, lowest_frequency,
                       highest_frequency, preemphasis_factor, window)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
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
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1+ (L/2)*numpy.sin(numpy.pi*n/L)
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
