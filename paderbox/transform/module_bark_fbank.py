import numpy as np
from nt.transform import stft, stft_to_spectrogram, offset_compensation, preemphasis
import scipy


def bark_fbank(time_signal, sample_rate=16000, window_length=400, stft_shift=160,
               number_of_filters=23, stft_size=512, lowest_frequency=0,
               highest_frequency=None, preemphasis_factor=0.97,
               window=scipy.signal.hamming, sum_power=True):
    """

    Compute Bark-filterbank energy features from an audio signal.

    Source: http://labrosa.ee.columbia.edu/matlab/rastamat/

    :param time_signal: the audio signal from which to compute the features.
        Should be a N*1 array.
    :param sample_rate: the sample rate of the time_signal.
    :param window_length: the length of the analysis window in samples.
        Default is 400 (25 milliseconds @ 16kHz).
    :param stft_shift: the step between successive windows in samples.
        Default is 160 (10 milliseconds @ 16kHz).
    :param number_of_filters: the number of filters in the bark filterbank,
        Default is 23 (one per bark).
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of bark filters.
        In Hz. Default is 0.
    :param highest_frequency: highest band edge of bark filters.
        In Hz. Default is samplerate/2.
    :param preemphasis_factor: apply preemphasis filter with preemphasis_factor
        as coefficient. 0 is no filter. Default is 0.97.
    :param window: window function used for stft. Default is hamming window.
    :param sum_power: whether to sqrt the power spectrum prior to calculate the
        features and transform them back with power of 2 or not.
        Default is True.
    :return: A numpy array with shape (frames by number_of_filters) containing
        the Bark features. Each row holds one feature vector.
    """

    #
    time_signal = offset_compensation(time_signal)
    time_signal = preemphasis(time_signal, preemphasis_factor)

    highest_frequency = highest_frequency or sample_rate/2

    # Compute power spectrogram
    stft_signal = stft(time_signal=time_signal, size=stft_size,
                       shift=stft_shift, window=window,
                       window_length=window_length, fading=False)
    spectrogram = stft_to_spectrogram(stft_signal)

    # Transform to bark scale
    nframes, nfreqs = spectrogram.shape
    filterbanks = get_bark_filterbanks(nfft=stft_size, sample_rate=sample_rate,
                                       number_of_filters=number_of_filters,
                                       lowest_frequency=lowest_frequency,
                                       highest_frequency=highest_frequency)
    filterbanks = filterbanks.T[0:nfreqs]

    if sum_power:
        feature = np.dot(np.sqrt(spectrogram), filterbanks) ** 2
    else:
        feature = np.dot(spectrogram, filterbanks)

    # if feat is zero, we get problems with log
    feature = np.where(feature == 0, np.finfo(float).eps, feature)

    return feature


def get_bark_filterbanks(nfft, sample_rate=16000, number_of_filters=23, width=1,
                         lowest_frequency=0, highest_frequency=8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark
    (Critical Bandwidth Analysis)

	While the matrix has nfft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(nfft,sampling_rate)*stft(xincols,nfft)

	:param nfft: source FFT size at sample rate sampling_rate.
	:param sample_rate: sample rate of the FFT signal. Default is 16000.
	:param number_of_filters: number of output bands required.
	    Default is 23 (one per bark).
 	:param width: constant width of each band in Bark. Default is 1.
	:param lowest_frequency: minimum frequency of FFT in Hz. Default is 0.
	:param highest_frequency: maximum frequency of FFT in Hz. Default is 8000.
    """

    min_bark = hz2bark(lowest_frequency)
    nyqbark = hz2bark(highest_frequency) - min_bark

    if (number_of_filters == 0):
        number_of_filters = int(np.ceil(nyqbark)) + 1

    # initialize matrix
    W = np.zeros((number_of_filters, nfft))

    # bark per filt
    step_barks = nyqbark / (number_of_filters - 1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(np.linspace(0, (nfft / 2),
                                   (nfft / 2) + 1) * sample_rate / nfft)

    for i in range(0, number_of_filters):
        f_bark_mid = min_bark + (i) * step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = binbarks - f_bark_mid - 0.5
        hif = binbarks - f_bark_mid + 0.5
        W[i, 0:(nfft // 2) + 1] = \
            10 ** (np.minimum(0, np.minimum(hif / width, lof * (-2.5 / width))))

    return W


def hz2bark(hz):
    return 6 * np.arcsinh(hz / 600)


def bark2hz(bark):
    return 600 * np.sinh(bark / 6)
