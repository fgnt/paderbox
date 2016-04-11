import numpy as np
from scipy.signal import lfilter
from nt.transform import stft_to_spectrogram, stft, istft, mfcc

def rasta_plp(time_signal, sample_rate=16000, modelorder = 8, do_rasta = True):
    """
    Compute PLP features from an audio signal with optional RASTA filtering (enabled by default).

    Ported from Matlab (Source: http://labrosa.ee.columbia.edu/matlab/rastamat/)
    :param time_signal: the audio signal from which to compute features.
    :param sample_rate: the samplerate of the signal we are working with
    :param do_rasta: enable RASTA-filtering
    :return: A numpy array containing features. Each row holds 1 feature vektor
    """
    # compute power spectrogram
    powerspec = stft_to_spectrogram(stft(time_signal))

    # transform to bark scale (Critical bandwidth analysis)
    nframes, nfreqs = powerspec.shape
    nfft = (nfreqs - 1)*2
    fft2bark_matrix = get_fft2bark_matrix(nfft, sample_rate, 23, 1, 0, sample_rate/2)
    fft2bark_matrix = fft2bark_matrix.T[0:nfreqs] # Second half is all zero and not needed. Transpose Matrix from Matlab
    aspectrum = np.dot(np.sqrt(powerspec), fft2bark_matrix)**2

    # do optional RASTA filtering
    if do_rasta:
        # in log domain
        aspectrum = np.where(aspectrum==0, np.finfo(float).eps, aspectrum)
        aspectrum = np.exp(filter_rasta(np.log(aspectrum)))

    # final auditory compressions
    postspectrum = postaud(aspectrum, sample_rate/2)

    # Linear predictive coding
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
    """Apply RASTA filtering to the input signal.

    :param x: the input audio signal to filter.
        default filter is single pole at 0.94
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
        y[i, :4], zf[i, :4] = lfilter(numerator, 1, x[i, :4], axis=-1, zi=[0, 0, 0, 0])

    # .. but don't keep any of these values, just output zero at the beginning
    y = np.zeros(x.shape)

    # Apply the full filter to the rest of the signal, append it
    for i in range(y.shape[0]):
        y[i, 4:] = lfilter(numerator, denominator, x[i, 4:], axis=-1, zi=zf[i, :])[0]

    return y.T

def get_fft2bark_matrix(nfft, sample_rate = 16000, nfilts=23, width = 1, minfreq = 0, maxfreq = 8000):
    """
    Generate a matrix of weights to combine FFT bins into Bark (Critical Bandwidth Analysis)

	While the matrix has nfft columns, the second half are all zero.
    Hence, Bark spectrum is fft2barkmx(nfft,sampling_rate)*stft(xincols,nfft)

	:param nfft: source FFT size at sampling rate sampling_rate
	:param sample_rate: sampling rate of the FFT signal
	:param nfilts: number of output bands required (default one per bark -> 23)
 	:param width: constant width of each band in Bark
	:param minfreq: minimum frequency of FFT in hz
	:param maxfeq: maximum frequency of FFT in hz
    """

    min_bark = hz2bark(minfreq)
    nyqbark = hz2bark(maxfreq) - min_bark

    if (nfilts == 0):
        nfilts = int(np.ceil(nyqbark))+1

    # initialize matrix
    W = np.zeros((nfilts, nfft))

    # bark per filt
    step_barks = nyqbark/(nfilts-1)

    # Frequency of each FFT bin in Bark
    binbarks = hz2bark(np.linspace(0,(nfft/2),(nfft/2)+1)*sample_rate/nfft)

    for i in range(0, nfilts):
        f_bark_mid = min_bark + (i)*step_barks
        # Linear slopes in log-space (i.e. dB) intersect to trapezoidal window
        lof = binbarks - f_bark_mid - 0.5
        hif = binbarks - f_bark_mid + 0.5
        W[i,0:(nfft/2)+1] = 10**(np.minimum(0, np.minimum(hif/width, lof * (-2.5/width))))

    return W

def hz2bark(hz):
    return 6 * np.arcsinh(hz/600)

def bark2hz(bark):
    return 600 * np.sinh(bark/6)

def postaud(x, highest_frequency):
    """
    Do loudness equalization and cube root compression.
    :param x: Critical band filters
    :param highest_frequency:
    :return:
    """
    nframes, nbands = x.shape

    # Include frequency points at extremes, discard later
    number_of_frequency_points = nbands

    bandcfhz = bark2hz(np.linspace(0, hz2bark(highest_frequency), number_of_frequency_points))

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
    y, e = levinson(r, modelorder)  # does not work..

    # Normalize each poly by gain
    y = y.T / np.tile(e.T, (modelorder + 1, 1))

    return y

    
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
    spec = spec.T   # Because Matlab switched cols/rows
    nrow, ncol = spec.shape

    # Make the DCT matrix
    dctm = np.zeros((ncep, nrow))
    for i in range(ncep):
        dctm[i,:] = np.cos((i-1)* np.linspace(1, (2*nrow - 1), (2*nrow - 1)/2 + 1)/(2*nrow)*np.pi)* np.sqrt(2/nrow)
    dctm[0] = dctm[0]/np.sqrt(2)

    return np.dot(dctm, np.log(spec)).T