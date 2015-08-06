"""
Provides ssc features.
"""

import numpy
from nt.transform.module_filter import offcomp
from nt.transform.module_filter import preemphasis
from nt.transform.module_stft import stft
from nt.transform.module_stft import stft_to_spectrogram
from nt.transform.module_fbank import get_filterbanks
import scipy.signal

def ssc(time_signal, sample_rate=16000, window_length=400, stft_shift=160,
        number_of_filters=26, stft_size=512,
        lowest_frequency=0, highest_frequency=None,
        preemphasis_factor=0.97, window=scipy.signal.hamming):
    """
    Compute Spectral Subband Centroid features from an audio signal.

    This is most likely broken.
    See this: https://maxwell.ict.griffith.edu.au/spl/publications/papers/icassp98_kkp_ssc.pdf

    Illustrations: http://ntjenkins.upb.de/view/PythonToolbox/job/python_toolbox_notebooks/HTML_Report/toolbox_examples/transform/06%20-%20Additional%20features.html


    :param time_signal: the audio signal from which to compute features.
        Should be an N*1 array
    :param sample_rate: the samplerate of the signal we are working with.
    :param window_length: the length of the analysis window in seconds.
        Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds.
        Default is 0.01s (10 milliseconds)
    :param number_of_filters: the number of filters in the filterbank,
        default 26.
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of mel filters. In Hz,
        default is 0.
    :param highest_frequency: highest band edge of mel filters.
        In Hz, default is samplerate/2
    :param preemphasis: apply preemphasis filter with preemph as coefficient.
        0 is no filter. Default is 0.97.
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features.
        Each row holds 1 feature vector.
    """
    highest_frequency = highest_frequency or sample_rate / 2

    time_signal = offcomp(time_signal)
    time_signal = preemphasis(time_signal, preemphasis_factor)

    stft_signal = stft(time_signal, size=stft_size, shift=stft_shift,
                            window=window, window_length=window_length)

    spectrogram = stft_to_spectrogram(stft_signal)

    # if things are all zeros we get problems
    pspec = numpy.where(spectrogram == 0, numpy.finfo(float).eps, spectrogram)

    fb = get_filterbanks(number_of_filters, stft_size, sample_rate,
                               lowest_frequency, highest_frequency)

    # compute the filterbank energies
    feat = numpy.dot(spectrogram, fb.T)
    R = numpy.tile(numpy.linspace(1, sample_rate / 2, numpy.size(pspec, 1)),
                   (numpy.size(pspec, 0), 1))

    return numpy.dot(pspec * R, fb.T) / feat
