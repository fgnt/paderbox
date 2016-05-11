import numpy as np
import scipy
from scipy.signal import lfilter
from nt.transform import bark_fbank
from nt.transform.module_bark_fbank import bark2hz, hz2bark

def rasta_plp(time_signal, sample_rate=16000, modelorder = 8, do_rasta = True,
              window_length=400, stft_shift=160, number_of_filters=23,
              stft_size = 512, lowest_frequency=0, highest_frequency=None,
              preempnasis_factor = 0, window=scipy.signal.hamming,
              sum_power=True):
    """
    Compute PLP features from an audio signal with optional RASTA
    (RelAtive SpecTrAl) filtering (enabled by default).

    Currently this code only supports a modelorder of 0 (which means no linear
    prediction is applied). It computes the bark_fbank features, does some
    auditory compressions and optional applies RASTA filtering.

    Ported from Matlab (Source: http://labrosa.ee.columbia.edu/matlab/rastamat/)

    :param time_signal: the audio signal from which to compute features.
    :param sample_rate: the samplerate of the signal we are working with.
        Default is 16000.
    :param modelorder: Order of the PLP Model. Currently only modelorder=0
        supported.
    :param do_rasta: enable RASTA-filtering. Default is True.
    :param window_length: the length of the analysis window in samples for
        critical band analysis.
        Default is 400 (25 milliseconds @ 16kHz)
    :param stft_shift: the step between successive windows in seconds.
        Default is 160 (10 milliseconds @ 16kHz)
    :param number_of_filters: the number of filters in the filterbank,
        Default is 23.
    :param stft_size: the FFT size. Default is 512.
    :param lowest_frequency: lowest band edge of bark filters.
        In Hz. Default is 0.
    :param highest_frequency: highest band edge of bark filters.
        In Hz. Default is samplerate/2
    :param preemphasis_factor: apply preemphasis filter with preemphasis_factor
        as coefficient.
        Default is 0: no filter. (As in Matlab)
    :param window: window function used for stft. Default is hamming window.
    :param sum_power: whether to sqrt the power spectrum prior to calculate the
        bark features and transform them back with power of 2 or not.
        Default is True.
    :return: A numpy array of shape (frames, number_of_filters) containing
        features. Each row holds 1 feature vektor.
    """

    highest_frequency = highest_frequency or sample_rate/2

    # Compute Citical Bands
    aspectrum = bark_fbank(time_signal, sample_rate = sample_rate,
                           window_length = window_length, window=window,
                           stft_size=stft_size, stft_shift=stft_shift,
                           number_of_filters = number_of_filters,
                           lowest_frequency=lowest_frequency,
                           highest_frequency=highest_frequency,
                           preemphasis_factor=preempnasis_factor,
                           sum_power=sum_power)

    # do optional RASTA filtering
    if do_rasta:
        # in log domain
        aspectrum = np.exp(filter_rasta(np.log(aspectrum)))

    # final auditory compressions
    postspectrum = postaud(aspectrum, highest_frequency)

    # Linear predictive coding
    modelorder = 0  # Ensure this doesn't get executed
    if modelorder > 0:  # This does not work yet
        # LPC analysis
        lpcas = linear_predictive_coding(postspectrum, modelorder)

        # Convert to cepstra
        cepstra = lpc2cep(lpcas, modelorder + 1)

        # or to spectra
        spectra = lpc2spec(lpcas, aspectrum.shape[1])
    else:
        # No LPC smolthing of spectrum
        spectra = postspectrum
        cepstra = spec2cep(spectra)

    return cepstra, spectra


def filter_rasta(x):
    """Apply RASTA filtering to the input signal. Default filter is single pole
    at 0.94

    :param x: the input audio signal to filter.
    :return: filtered signal
    """

    x = x.T

    numerator = np.arange(.2, -.3, -.1)
    denominator = np.array([1, -0.94])

    # Initialize the state.  This avoids a big spike at the beginning
    # resulting from the dc offset level in each band.
    # specify the FIR part to get the state right (but not the IIR part)
    y = np.zeros(x.shape)
    zf = np.zeros((x.shape[0], 4))
    for i in range(y.shape[0]):
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4],
                                      axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = np.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:],
                           axis=-1, zi=zf[i, :])[0]

    return y.T

def postaud(x, highest_frequency):
    """
    Do loudness equalization (Hynek) and cube root compression.

    :param x: bark spectrum to equalize and compress.
    :param highest_frequency: highest band edge of bark filters.
    :return: np array of same shape as x containing equalized and compressed
        bark spectrum.
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    number_of_frequency_points = nbands

    bandcfhz = bark2hz(np.linspace(0, hz2bark(highest_frequency),
                                   number_of_frequency_points))

    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz**2
    ftmp = fsq + 1.6e5
    eql = ((fsq/ftmp)**2) * ((fsq + 1.44e6) / (fsq + 9.61e6))

    # weight the critical bands
    z = np.tile(eql, (nframes, 1)) * x

    # cube root compress
    z = z ** 0.33

    # replicate first and last band
    z = z.T
    z = np.concatenate([[z[1]], z[1:nbands-1], [z[nbands-1]]])

    return z.T

def linear_predictive_coding(x, modelorder = 8):
    nframes, nbands = x.shape

    # Calculate autocorrelation
    x = x.T
    x = np.concatenate([x, np.flipud(x)[1:nbands]])

    r = np.zeros((x.shape))
    for i in range(x.shape[1]):
        r[:, i] = np.fft.ifft(x[:, i])

    # First half only
    r = r[1:nbands]

    # Find LPC coefficients by Levinson-Durbin
    #y, e = levinson(r, modelorder)  # does not work..

    # Normalize each poly by gain
    #y = y.T / np.tile(e.T, (modelorder + 1, 1))

    #return y
    return 0

    
def levinson(r, modelorder=8):
    """

    Source: https://github.com/cournape/talkbox/blob/master/scikits/talkbox/linpred/py_lpc.py

    :param r:
    :param modelorder:
    :return:
    """

    # Estimated coefficients
    a = np.empty(modelorder+1, r.dtype)
    # temporary array
    t = np.empty(modelorder+1, r.dtype)
    # Reflection coefficients
    k = np.empty(modelorder, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, modelorder+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(modelorder):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

def lpc2cep(a, nout):
    """
    Convert the LPC 'a' coefficients in each column of lpcas into frames of cepstra.
    :param a:
    :param nout: number of cepstra to produce
    :return:
    """
    nin, ncol = a.shape
    order = nin - 1

    nout = nout or order + 1

    c = np.zeros((nout, ncol))

    # First cep is log(Error) from Durbin
    c[1] = -np.log(a[1])

    # Renormalize lpc A coeffs
    a = a / np.tile(a[0], (nin, 1))

    for n in range(1, nout):
        sum = 0;
        for m in range(1, n):
            sum += ((n-m) * a[m]) * c[n-m]
        c[n] = -(a[n] + sum/(n))

    return c

def lpc2spec(lpcas, nout):
    rows, cols = lpcas.shape
    order = rows - 1

    gg = lpcas[1]
    aa = lpcas / np.tile(gg, (rows, 1))

    # Calculate the actual z-plane polyvals: nout points around unit circle
    zz = np.exp((-1j*np.linspace(0, nout-1, nout).T*np.pi/(nout-1))*np.linspace(0, order, order + 1))

    # Actual polyvals, in power (mag²)
    features = ((1 / np.abs(zz*aa))**2) / np.tile(gg, (nout, 1))

    F = np.zeros(cols, np.floor(rows/2))
    M = F

    for c in range(cols):
        aaa = aa[:][c]
        rr = np.roots(aaa.T)
        ff = np.angle(rr.T)
        zz = np.exp(1j*ff.T * np.linspace(0, len(aaa)-1, len(aaa)))
        mags = np.sqrt(((1 / np.abs(zz*aaa))**2)/gg[c]).T

        ix = np.argsort(ff)
        keep = ff[ix] > 0
        ix = ix[keep]
        F[c, 1:len(ix)] = ff(ix)
        M[c, 1:len(ix)] = mags(ix)

    return features, F, M

def spec2cep(spec, ncep = 13):
    """
    Calculate cepstra from spectral samples. This one does type 2 dct.

    :param spec: spectral samples
    :param ncep: count of cepstral coefficients per frame
    :return:
    """
    spec = spec.T   # Because Matlab switched cols/rows
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = np.zeros((ncep, nrow))
    for i in range(ncep):
        dctm[i,:] = np.cos(i* np.linspace(1, (2*nrow - 1), (2*nrow - 1)/2 + 1)/(2*nrow)*np.pi)* np.sqrt(2/nrow)
    dctm[0] = dctm[0]/np.sqrt(2)

    return np.dot(dctm, np.log(spec)).T