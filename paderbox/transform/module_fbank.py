"""
Provides fbank features and the fbank filterbank.
"""

from typing import Optional, Union, Callable

from cached_property import cached_property
import numpy as np
import scipy.signal

from paderbox.transform.module_filter import preemphasis_with_offset_compensation
from paderbox.transform.module_stft import stft
from paderbox.transform.module_stft import stft_to_spectrogram
import dataclasses


# pylint: disable=too-many-arguments,line-too-long

class MelTransform:
    def __init__(
            self,
            sample_rate: int,
            stft_size: int,
            number_of_filters: int,
            lowest_frequency: Optional[float] = 50,
            highest_frequency: Optional[float] = None,
            htk_mel=True,
            log: bool = True,
            eps: float = 1e-18,
            *,
            warping_fn: Optional[Callable] = None,
            independent_axis: tuple = (0,),
    ):
        """Transforms linear spectrogram to (log) mel spectrogram.

        Args:
            number_of_filters: number of filters to be applied
            sample_rate: sample rate of audio signal
            stft_size: fft_length used in stft
            lowest_frequency: onset of first filter
            highest_frequency: offset of last filter
            htk_mel: If True use HTK's hz to mel conversion definition else use
                Slaney's definition (cf. librosa.mel_frequencies doc).
            log: apply log to mel spectrogram
            eps:
            warping_fn: function to (randomly) remap fbank center frequencies
            independent_axis: independent axis for which independently warped
                filter banks are used.

        >>> sample_rate = 16000
        >>> highest_frequency = sample_rate/2
        >>> mel_transform = MelTransform(sample_rate, 512, 40, highest_frequency=highest_frequency)
        >>> spec = np.zeros((3, 1, 100, 257))
        >>> logmelspec = mel_transform(spec)
        >>> logmelspec.shape
        (3, 1, 100, 40)
        >>> rec = mel_transform.inverse(logmelspec)
        >>> rec.shape
        (3, 1, 100, 257)
        >>> from paderbox.utils.random_utils import Uniform
        >>> warping_fn = HzWarping(\
                warp_factor_sampling_fn=Uniform(low=.9, high=1.1),\
                boundary_frequency_ratio_sampling_fn=Uniform(low=.6,high=.7),\
                highest_frequency=highest_frequency,\
            )
        >>> mel_transform = MelTransform(sample_rate, 512, 40, warping_fn=warping_fn)
        >>> mel_transform(spec).shape
        (3, 1, 100, 40)
        >>> mel_transform = MelTransform(16000, 512, 40, warping_fn=warping_fn, independent_axis=(0,1,2))
        >>> mel_transform(spec).shape
        (3, 1, 100, 40)
        """
        self.sample_rate = sample_rate
        self.stft_size = stft_size
        self.number_of_filters = number_of_filters
        self.lowest_frequency = lowest_frequency
        self.highest_frequency = highest_frequency
        self.htk_mel = htk_mel
        self.log = log
        self.eps = eps
        self.warping_fn = warping_fn
        self.independent_axis = tuple(
            [independent_axis] if np.isscalar(independent_axis)
            else independent_axis
        )

    @cached_property
    def fbanks(self):
        """Create filterbank matrix according to member variables."""
        fbanks = get_fbanks(
            sample_rate=self.sample_rate,
            stft_size=self.stft_size,
            number_of_filters=self.number_of_filters,
            lowest_frequency=self.lowest_frequency,
            highest_frequency=self.highest_frequency,
            htk_mel=self.htk_mel,
        )
        fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + self.eps)
        return fbanks.T

    @cached_property
    def ifbanks(self):
        """Create (pseudo)-inverse of filterbank matrix."""
        return np.linalg.pinv(self.fbanks.T).T

    def __call__(self, x: np.ndarray):
        if self.warping_fn is None:
            x = x @ self.fbanks
        else:
            independent_axis = [ax if ax >= 0 else x.ndim+ax for ax in self.independent_axis]
            assert all([0 <= ax < x.ndim-1 for ax in independent_axis]), self.independent_axis
            size = [
                x.shape[i] if i in independent_axis else 1
                for i in range(x.ndim-1)
            ]
            fbanks = get_fbanks(
                sample_rate=self.sample_rate,
                stft_size=self.stft_size,
                number_of_filters=self.number_of_filters,
                lowest_frequency=self.lowest_frequency,
                highest_frequency=self.highest_frequency,
                htk_mel=self.htk_mel,
                warping_fn=self.warping_fn,
                size=tuple(size),
            ).astype(np.float32)
            fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + self.eps)
            fbanks = fbanks.swapaxes(-2, -1)
            # The following is the same as `np.einsum('...F,...FN->...N', x, fbanks)`, but much faster (see https://github.com/fgnt/paderbox/pull/35).
            if fbanks.shape[-3] == 1:
                x = x @ fbanks.squeeze(-3)
            else:
                x = (x[..., None, :] @ fbanks).squeeze(-2)
        if self.log:
            x = np.log(x + self.eps)
        return x

    def inverse(self, x: np.ndarray):
        """Invert the mel-filterbank transform."""
        if self.log:
            x = np.exp(x)
        return np.maximum(np.dot(x, self.ifbanks), 0.)


def get_fbanks(
        sample_rate: int, stft_size: int, number_of_filters: int,
        lowest_frequency: float = 0.,
        highest_frequency: Optional[float] = None,
        htk_mel=True,
        warping_fn: Optional[Callable] = None,
        size: tuple = ()
):
    """Computes mel filter banks

    Args:
        sample_rate:
        stft_size:
        number_of_filters: number of mel filter banks
        lowest_frequency: onset frequency of the first filter
        highest_frequency: offset frequency of the last filter
        htk_mel: If True use HTK's hz to mel conversion definition else use
            Slaney's definition (cf. librosa.mel_frequencies doc).
        warping_fn: optional function to warp the filter center frequencies,
            e.g., VTLP (https://www.cs.utoronto.ca/~hinton/absps/perturb.pdf)
        size: size of independent dims in front of filter bank dims, E.g., for
            size=(2,3) returned filterbanks have shape (*size, number_of_filters, stft_size//2+1).
            This is required when different warping is to be applied for
            different independent axis.

    Returns: array of filters

    >>> sample_rate = 8000
    >>> highest_frequency = sample_rate / 2
    >>> fbanks = get_fbanks(sample_rate, 32, 10)
    >>> fbanks[[0,-1]]
    array([[0.47080041, 0.52919959, 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.1996357 , 0.59750853, 0.99538136, 0.66925631,
            0.33462816, 0.        ]])
    >>> get_fbanks(sample_rate, 32, 10, warping_fn=HzWarping(\
            warp_factor_sampling_fn=lambda size: 0.9+0.2*np.random.rand(*size),\
            boundary_frequency_ratio_sampling_fn=lambda n: 0.7,\
            highest_frequency=highest_frequency,\
        )).shape
    (10, 17)
    >>> get_fbanks(sample_rate, 32, 10, size=(2,3), warping_fn=HzWarping(\
            warp_factor_sampling_fn=lambda size: 0.9+0.2*np.random.rand(*size),\
            boundary_frequency_ratio_sampling_fn=lambda n: 0.7,\
            highest_frequency=highest_frequency,\
        )).shape
    (2, 3, 10, 17)

    """
    highest_frequency = sample_rate / 2 if highest_frequency is None else highest_frequency
    if highest_frequency < 0:
        highest_frequency = highest_frequency % sample_rate / 2
    f = mel2hz(
        np.linspace(
            hz2mel(lowest_frequency, htk_mel=htk_mel),
            hz2mel(highest_frequency, htk_mel=htk_mel),
            number_of_filters + 2
        ),
        htk_mel=htk_mel,
    )
    if warping_fn is not None:
        f = warping_fn(f, size=size)
    k = hz2bin(f, sample_rate, stft_size)
    centers = k[..., 1:-1, None]
    onsets = np.minimum(k[..., :-2, None], centers - 1)
    offsets = np.maximum(k[..., 2:, None], centers + 1)
    idx = np.arange(stft_size // 2 + 1)
    fbanks = np.maximum(
        np.minimum(
            (idx-onsets)/(centers-onsets),
            (idx-offsets)/(centers-offsets)
        ),
        0
    )
    return np.broadcast_to(fbanks, (*size, *fbanks.shape[-2:]))


def hz2mel(frequency: Union[float, np.ndarray], htk_mel=True):
    """Convert frequencies in Hertz to Mel.

    !!! Copy of librosa.hz_to_mel !!!

    Args:
        frequency: a value in Hz. This can also be a numpy array, conversion
            proceeds element-wise.
        htk_mel: If True use HTK's hz to mel conversion definition else use
            Slaney's definition (cf. librosa.mel_frequencies doc).

    Returns: a value in Mels. If an array was passed in, an identical sized
        array is returned.

    """
    if htk_mel:
        return 2595 * np.log10(1 + frequency / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequency - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if not np.isscalar(frequency) and frequency.ndim:
        # If we have array data, vectorize
        log_t = frequency >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequency[log_t] / min_log_hz) / logstep
    elif frequency >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequency / min_log_hz) / logstep
    return mels


def mel2hz(frequency: Union[float, np.ndarray], htk_mel=True):
    """Convert frequencies in Mel to Hertz

    !!! Copy of librosa.mel_to_hz !!!

    Args:
        frequency: a value in Mels. This can also be a numpy array, conversion
            proceeds element-wise.
        htk_mel: If True use HTK's hz to mel conversion definition else use
            Slaney's definition (cf. librosa.mel_frequencies doc).

    Returns: a value in Hz. If an array was passed in, an identical sized
        array is returned.

    """
    if htk_mel:
        return 700 * (10 ** (frequency / 2595.0) - 1)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * frequency

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if not np.isscalar(frequency) and frequency.ndim:
        # If we have vector data, vectorize
        log_t = frequency >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (frequency[log_t] - min_log_mel))
    elif frequency >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (frequency - min_log_mel))
    return freqs


def bin2hz(
        fft_bin_index: Union[int, np.ndarray],
        sample_rate: int, stft_size: int
):
    """Convert fft bin indices to frequencies in Hz

    Args:
        fft_bin_index: fft bin index. This can also be a numpy array,
            conversion proceeds element-wise.
        sample_rate:
        stft_size:

    Returns:

    """
    return sample_rate * fft_bin_index / stft_size


def hz2bin(
        frequency: Union[float, np.ndarray],
        sample_rate: int,
        stft_size: int
):
    """Convert frequencies in Hz to fft bin indices (soft, i.e. return value is a float not an int)

    Args:
        frequency: a value in Hz. This can also be a numpy array, conversion
            proceeds element-wise.
        sample_rate:
        stft_size:

    Returns:

    """
    return frequency * stft_size / sample_rate


def hz_warping(
        frequency: Union[float, np.ndarray],
        warp_factor: Union[float, np.ndarray],
        boundary_frequency_ratio: Union[float, np.ndarray],
        highest_frequency: float
):
    """Performs piece wise linear warping of frequencies in Hz.
    http://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf

    Args:
        frequency: frequency scalar or array in Hz
        warp_factor: scalar or array of warp_factors
        boundary_frequency_ratio: scalar or array of ratios such that
            boundary_frequency = boundary_frequency_ratio * sample_rate/2.
            Ratios have to be > 0.0 . If it is >= 1. the whole spectrogram is
            stretched or squeezed, i.e., a simple linear warping (not piecewise)
            is performed with warping(highest_frequency) != highest_frequency.
            Note that when used for VTLP with boundary_frequency_ratio > 1 and
            warp_factor > 1 frequencies may be mapped to frequencies beyond
            sample_rate / 2 (highest frequency in the stft) which would yield
            zeros in the mel frequency bands concerned.
        highest_frequency:

    Returns:

    >>> sample_rate = 16000
    >>> lowest_frequency = 0
    >>> highest_frequency = sample_rate/2
    >>> number_of_filters = 10
    >>> frequency = mel2hz(np.linspace(hz2mel(lowest_frequency), hz2mel(highest_frequency), number_of_filters+2))
    >>> frequency
    array([   0.        ,  180.21928115,  406.83711843,  691.7991039 ,
           1050.12629534, 1500.70701371, 2067.29249375, 2779.74887082,
           3675.63149949, 4802.16459006, 6218.73051459, 8000.        ])
    >>> f_warped = hz_warping(frequency, warp_factor=1.1, boundary_frequency_ratio=.6, highest_frequency=highest_frequency)
    >>> f_warped
    array([   0.        ,  198.24120926,  447.52083027,  760.97901429,
           1155.13892487, 1650.77771509, 2274.02174313, 3057.7237579 ,
           4043.19464944, 5185.90483925, 6432.48285284, 8000.        ])
    >>> warp_factors = np.array([.9, 1.1])
    >>> f_warped = hz_warping(frequency, warp_factor=warp_factors, boundary_frequency_ratio=.6, highest_frequency=highest_frequency)
    >>> f_warped
    array([[   0.        ,  162.19735303,  366.15340659,  622.61919351,
             945.11366581, 1350.63631234, 1860.56324438, 2501.77398374,
            3308.06834954, 4322.48927857, 5951.54009178, 8000.        ],
           [   0.        ,  198.24120926,  447.52083027,  760.97901429,
            1155.13892487, 1650.77771509, 2274.02174313, 3057.7237579 ,
            4043.19464944, 5185.90483925, 6432.48285284, 8000.        ]])
    >>> f_warped/frequency
    array([[       nan, 0.9       , 0.9       , 0.9       , 0.9       ,
            0.9       , 0.9       , 0.9       , 0.9       , 0.90011269,
            0.95703457, 1.        ],
           [       nan, 1.1       , 1.1       , 1.1       , 1.1       ,
            1.1       , 1.1       , 1.1       , 1.1       , 1.07990985,
            1.03437234, 1.        ]])
    >>> warp_factors = warp_factors[..., None]
    >>> hz_warping(\
        frequency, warp_factor=warp_factors, boundary_frequency_ratio=.75,\
        highest_frequency=highest_frequency,\
    ).shape
    (2, 1, 12)
    >>> f_warped = hz_warping(\
        4000, warp_factor=warp_factors, boundary_frequency_ratio=.75,\
        highest_frequency=highest_frequency,\
    )
    >>> f_warped, f_warped.shape
    (array([[3600.],
           [4400.]]), (2, 1))
    >>> f_warped = hz_warping(\
        4000, warp_factor=.9, boundary_frequency_ratio=.75,\
        highest_frequency=highest_frequency,\
    )
    >>> f_warped, f_warped.shape
    (3600.0, ())
    >>> f_warped = hz_warping(\
        8000, warp_factor=.9, boundary_frequency_ratio=np.array([.75, 1.]),\
        highest_frequency=highest_frequency,\
    )
    >>> f_warped, f_warped.shape
    (array([8000., 7200.]), (2,))
    """
    frequency = np.array(frequency)
    assert (0 <= frequency).all() and (frequency <= (highest_frequency + 1e-6)).all(), (
        np.min(frequency), np.max(frequency), highest_frequency
    )
    warp_factor = np.array(warp_factor)
    assert (warp_factor > 0).all(), warp_factor
    boundary_frequency_ratio = np.array(boundary_frequency_ratio)
    assert (boundary_frequency_ratio > 0).all(), warp_factor
    boundary_frequency = boundary_frequency_ratio * highest_frequency
    breakpoints = boundary_frequency * np.minimum(warp_factor, 1) / warp_factor

    if breakpoints.ndim == 0:
        breakpoints = np.array(breakpoints)
    breakpoints[
        (breakpoints > highest_frequency)
        + ((warp_factor * breakpoints) > highest_frequency)
    ] = highest_frequency
    bp_value = warp_factor * breakpoints

    for _ in range(frequency.ndim):
        warp_factor = warp_factor[..., None]
        breakpoints = breakpoints[..., None]
        bp_value = bp_value[..., None]
    frequency, breakpoints, bp_value = np.broadcast_arrays(
        frequency, breakpoints, bp_value
    )
    f_warped_first_piece = warp_factor * frequency
    f_warped_second_piece = (
            bp_value
            + (
                    (frequency - breakpoints)  # <= 0
                    * (highest_frequency - bp_value)
                    / np.maximum(highest_frequency - breakpoints, 1e-18)  # breakpoints <= highest_frequency
            )
    )
    f_warped = (
        np.minimum(f_warped_first_piece, f_warped_second_piece) * (warp_factor >= 1.)
        + np.maximum(f_warped_first_piece, f_warped_second_piece) * (warp_factor < 1.)
    )
    return f_warped


def mel_warping(
        frequency: Union[float, np.ndarray],
        warp_factor: Union[float, np.ndarray],
        boundary_frequency_ratio: Union[float, np.ndarray],
        highest_frequency: float,
        htk_mel=True,
):
    """Transforms frequency to Mel domain and performs piecewise linear warping
    there. Finally transforms warped frequency back to Hz.

    Args:
        frequency:
        warp_factor:
        boundary_frequency_ratio:
        highest_frequency:

    Returns:

    """
    frequency = hz2mel(frequency, htk_mel=htk_mel)
    if highest_frequency is not None:
        highest_frequency = hz2mel(highest_frequency, htk_mel=htk_mel)
    frequency = hz_warping(frequency, warp_factor, boundary_frequency_ratio, highest_frequency)
    return mel2hz(frequency, htk_mel=htk_mel)


@dataclasses.dataclass
class HzWarping:
    """
    >>> sample_rate = 16000
    >>> lowest_frequency = 0
    >>> highest_frequency = sample_rate/2
    >>> number_of_filters = 10
    >>> f = mel2hz(np.linspace(hz2mel(lowest_frequency), hz2mel(highest_frequency), number_of_filters+2))
    >>> from paderbox.utils.random_utils import Uniform
    >>> warping_fn = HzWarping(\
            warp_factor_sampling_fn=Uniform(low=.9, high=1.1),\
            boundary_frequency_ratio_sampling_fn=Uniform(low=.6,high=.7),\
            highest_frequency=highest_frequency,\
        )
    >>> np.random.seed(0)
    >>> warping_fn(f)/f
    array([      nan, 1.0097627, 1.0097627, 1.0097627, 1.0097627, 1.0097627,
           1.0097627, 1.0097627, 1.0097627, 1.0097627, 1.0055517, 1.       ])
    >>> np.random.seed(1)
    >>> warping_fn(f)/f
    array([       nan, 0.9834044 , 0.9834044 , 0.9834044 , 0.9834044 ,
           0.9834044 , 0.9834044 , 0.9834044 , 0.9834044 , 0.9834044 ,
           0.99025952, 1.        ])
    """
    warp_factor_sampling_fn: Callable
    boundary_frequency_ratio_sampling_fn: Callable
    highest_frequency: float

    def __call__(self, frequency: Union[float, np.ndarray], size: tuple = ()):
        return hz_warping(
            frequency,
            warp_factor=self.warp_factor_sampling_fn(size),
            boundary_frequency_ratio=self.boundary_frequency_ratio_sampling_fn(size),
            highest_frequency=self.highest_frequency,
        )


class MelWarping(HzWarping):
    def __call__(self, frequency: Union[float, np.ndarray], size: tuple = ()):
        return mel_warping(
            frequency,
            warp_factor=self.warp_factor_sampling_fn(size),
            boundary_frequency_ratio=self.boundary_frequency_ratio_sampling_fn(size),
            highest_frequency=self.highest_frequency,
        )


def fbank(
        time_signal: np.ndarray,
        sample_rate: int = 16000,
        window_length: int = 400,
        stft_shift: int = 160,
        number_of_filters: int = 23,
        stft_size: int = 512,
        lowest_frequency: float = 0.,
        highest_frequency: Optional[float] = None,
        preemphasis_factor: float = 0.97,
        window: Callable = scipy.signal.windows.hamming,
        denoise: bool = False
):
    """Compute Mel-filterbank energy features from an audio signal.

    Source: https://github.com/jameslyons/python_speech_features
    Tutorial: http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/ # noqa

    Illustrations: http://ntjenkins.upb.de/view/PythonToolbox/job/python_toolbox_notebooks/HTML_Report/toolbox_examples/transform/06%20-%20Additional%20features.html

    Args:
        time_signal: the audio signal from which to compute features.
            Should be an N*1 array
        sample_rate: the sample rate of the signal we are working with.
        window_length: the length of the analysis window in samples.
            Default is 400 (25 milliseconds @ 16kHz)
        stft_shift: the step between successive windows in samples.
            Default is 160 (10 milliseconds @ 16kHz)
        number_of_filters: the number of filters in the filterbank, default 23.
        stft_size: the FFT size. Default is 512.
        lowest_frequency: lowest band edge of mel filters.
            In Hz, default is 0.
        highest_frequency: highest band edge of mel filters.
            In Hz, default is samplerate/2
        preemphasis_factor: apply preemphasis filter with preemph as coefficient.
            0 is no filter. Default is 0.97.
        window: window function used for stft
        denoise:

    Returns: A numpy array of size (frames by number_of_filters) containing the
        Mel filterbank features.

    """
    highest_frequency = highest_frequency or sample_rate / 2
    time_signal = preemphasis_with_offset_compensation(
        time_signal, preemphasis_factor)

    stft_signal = stft(
        time_signal,
        size=stft_size, shift=stft_shift,
        window=window, window_length=window_length,
        fading=None
    )

    spectrogram = stft_to_spectrogram(stft_signal) / stft_size

    mel_transform = MelTransform(
        sample_rate=sample_rate,
        stft_size=stft_size,
        number_of_filters=number_of_filters,
        lowest_frequency=lowest_frequency,
        highest_frequency=highest_frequency,
        log=False
    )
    feature = mel_transform(spectrogram)

    if denoise:
        feature -= np.min(feature, axis=0)

    return feature


def logfbank(
        time_signal: np.ndarray,
        sample_rate: int = 16000,
        window_length: int = 400,
        stft_shift: int = 160,
        number_of_filters: int = 23,
        stft_size: int = 512,
        lowest_frequency: float = 0.,
        highest_frequency: Optional[float] = None,
        preemphasis_factor: float = 0.97,
        window: Callable = scipy.signal.windows.hamming,
        denoise: bool = False,
        eps: float = 1e-18,
):
    """Generates log fbank features from time signal.

    Simply wraps fbank function. See parameters there.
    """
    return np.log(
        fbank(
            time_signal,
            sample_rate=sample_rate,
            window_length=window_length,
            stft_shift=stft_shift,
            number_of_filters=number_of_filters,
            stft_size=stft_size,
            lowest_frequency=lowest_frequency,
            highest_frequency=highest_frequency,
            preemphasis_factor=preemphasis_factor,
            window=window,
            denoise=denoise
        ) + eps
    )
