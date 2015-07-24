import numpy
import nt.transform.filter as filter
import nt.transform.stft as stft
import scipy.signal

def fbank(signal, samplerate=16000, winlen=400, winstep=160,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
          winfunc=scipy.signal.hamming):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features.
        Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds.
        Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds.
        Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters.
        In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient.
        0 is no filter. Default is 0.97.
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt)
        containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame
        (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    signal = filter.offcomp(signal)
    signal = filter.preemphasis(signal, preemph)

    stft_signal = stft.stft(signal, size=nfft, shift=winstep,
                      window=winfunc, window_length=winlen)

    spectrogram = stft.stft_to_spectrogram(stft_signal)

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)

    # compute the filterbank energies
    feat = numpy.dot(spectrogram, fb.T)

    # if feat is zero, we get problems with log
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)

    return feat


def get_filterbanks(number_of_filters=20, nfft=1024, samplerate=16000,
                    lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns
    correspond to fft bins. The filters are returned as an array of size
    nfilt * (nfft/2 + 1)

    Source: https://github.com/jameslyons/python_speech_features

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 1024.
    :param samplerate: the samplerate of the signal we are working with.
        Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel, highmel, number_of_filters+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = numpy.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = numpy.zeros([number_of_filters, nfft/2+1])
    for j in range(0, number_of_filters):
        for i in range(int(bin[j]),int(bin[j+1])):
            fbank[j,i] = (i - bin[j])/(bin[j+1]-bin[j])
        for i in range(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion
        proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized
        array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.0)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion
        proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized
        array is returned.
    """
    return 700*(10**(mel/2595.0)-1)
