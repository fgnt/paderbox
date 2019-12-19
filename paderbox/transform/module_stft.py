"""
This file contains the STFT function and related helper functions.
"""
import string
import typing
from math import ceil

import numpy as np
from numpy.fft import rfft, irfft
from scipy import signal

from paderbox.array import roll_zeropad
from paderbox.array import segment_axis
from paderbox.utils.mapping import Dispatcher


def stft(
        time_signal,
        size: int = 1024,
        shift: int = 256,
        *,
        axis=-1,
        window: [str, typing.Callable] = signal.windows.blackman,
        window_length: int = None,
        fading: typing.Optional[typing.Union[bool, str]] = 'full',
        pad: bool = True,
        symmetric_window: bool = False,
) -> np.array:
    """
    ToDo: Open points:
     - sym_window need literature
     - fading why it is better?
     - should pad have more degrees of freedom?

    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    :param time_signal: Multi channel time signal with dimensions
        AA x ... x AZ x T x BA x ... x BZ.
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift, the step between successive frames in
        samples. Typically shift is a fraction of size.
    :param axis: Scalar axis of time.
        Default: None means the biggest dimension.
    :param window: Window function handle. Default is windows.blackman window.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :param pad: If true zero pad the signal to match the shape, else cut
    :param symmetric_window: symmetric or periodic window. Assume window is
        periodic. Since the implementation of the windows in scipy.signal have a
        curious behaviour for odd window_length. Use window(len+1)[:-1]. Since
        is equal to the behaviour of MATLAB.
    :return: Single channel complex STFT signal with dimensions
        AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    time_signal = np.asarray(time_signal)

    axis = axis % time_signal.ndim

    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [False, None]:
        pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int)
        if fading == 'half':
            pad_width[axis, 0] = (window_length - shift) // 2
            pad_width[axis, 1] = ceil((window_length - shift) / 2)
        else:
            pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    window = _get_window(
        window=window,
        symmetric_window=symmetric_window,
        window_length=window_length,
    )

    time_signal_seg = segment_axis(
        time_signal,
        window_length,
        shift=shift,
        axis=axis,
        end='pad' if pad else 'cut'
    )

    letters = string.ascii_lowercase[:time_signal_seg.ndim]
    mapping = letters + ',' + letters[axis + 1] + '->' + letters

    try:
        # ToDo: Implement this more memory efficient
        return rfft(
            np.einsum(mapping, time_signal_seg, window),
            n=size,
            axis=axis + 1,
        )
    except ValueError as e:
        raise ValueError(
            f'Could not calculate the stft, something does not match.\n'
            f'mapping: {mapping}, '
            f'time_signal_seg.shape: {time_signal_seg.shape}, '
            f'window.shape: {window.shape}, '
            f'size: {size}'
            f'axis+1: {axis+1}'
        ) from e


def stft_with_kaldi_dimensions(
        time_signal,
        size: int = 512,
        shift: int = 160,
        *,
        axis=-1,
        window=signal.windows.blackman,
        window_length=400,
        symmetric_window: bool = False
):
    """
    The Kaldi implementation uses another non standard window.
    See:
     - https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.h#L48
     - https://github.com/kaldi-asr/kaldi/blob/81b7a1947fb8df501a2bbb680b65ce18ce606cff/src/feat/feature-window.h#L48

    ..note::
       Kaldi uses symmetric_window == True
        - https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-window.cc#L113

    """
    # ToDo: Write test to force this function to fulfill the old kaldi_dims
    #       argument
    # if kaldi_dims:
    #     nsamp = time_signal.shape[axis]
    #     frames = time_signal_seg.shape[axis]
    #     expected_frames = 1 + ((nsamp - size) // shift)
    #     if frames != expected_frames:
    #         raise ValueError('Expected {} frames, got {}'.format(
    #             expected_frames, frames))
    return stft(
        time_signal,
        size=size,
        shift=shift,
        axis=axis,
        window=window,
        window_length=window_length,
        fading=None,
        pad=False,
        symmetric_window=symmetric_window
    )


_window_dispatcher = Dispatcher({
    'blackman': signal.windows.blackman,
    'hann': signal.windows.hann,
    'boxcar': signal.windows.boxcar,
    'triang': signal.windows.triang,
    'hamming':  signal.windows.hamming,
    'parzen': signal.windows.parzen,
    'cosine': signal.windows.cosine,
    'blackmanharris': signal.windows.blackmanharris,
    'flattop': signal.windows.flattop,
    'tukey': signal.windows.tukey,
    'bartlett': signal.windows.bartlett,
    'bohman': signal.windows.bohman,
    # 'kaiser2': functools.partial(signal.windows.kaiser, beta=2),
    # 'kaiser3': functools.partial(signal.windows.kaiser, beta=2),
})


def _get_window(window, symmetric_window, window_length):
    """Returns the window.

    Args:
        window: callable or str
        symmetric_window:
        window_length:

    Returns:
        1D Array of length window_length.

    >>> _get_window('hann', False, 4)  # common stft window
    array([0. , 0.5, 1. , 0.5])
    >>> _get_window('hann', True, 4)  # uncommon stft window, common for filter
    array([0.  , 0.75, 0.75, 0.  ])
    """

    if callable(window):
        pass
    elif isinstance(window, str):
        window = _window_dispatcher[window]
    else:
        raise TypeError(window)

    if symmetric_window:
        window = window(window_length)
    else:
        # https://github.com/scipy/scipy/issues/4551
        window = window(window_length + 1)[:-1]

    return window


def _samples_to_stft_frames(
        samples,
        size,
        shift,
        *,
        pad=True,
        fading=None,
):
    """
    Calculates number of STFT frames from number of samples in time domain.

    Args:
        samples: Number of samples in time domain.
        size: FFT size.
            window_length often equal to FFT size. The name size should be
            marked as deprecated and replaced with window_length.
        shift: Hop in samples.
        pad: See stft.
        fading: See stft. Note to keep old behavior, default value is False.

    Returns:
        Number of STFT frames.

    >>> _samples_to_stft_frames(19, 16, 4)
    2
    >>> _samples_to_stft_frames(20, 16, 4)
    2
    >>> _samples_to_stft_frames(21, 16, 4)
    3

    >>> stft(np.zeros(19), 16, 4, fading=None).shape
    (2, 9)
    >>> stft(np.zeros(20), 16, 4, fading=None).shape
    (2, 9)
    >>> stft(np.zeros(21), 16, 4, fading=None).shape
    (3, 9)

    >>> _samples_to_stft_frames(19, 16, 4, fading='full')
    8
    >>> _samples_to_stft_frames(20, 16, 4, fading='full')
    8
    >>> _samples_to_stft_frames(21, 16, 4, fading='full')
    9

    >>> stft(np.zeros(19), 16, 4).shape
    (8, 9)
    >>> stft(np.zeros(20), 16, 4).shape
    (8, 9)
    >>> stft(np.zeros(21), 16, 4).shape
    (9, 9)

    >>> _samples_to_stft_frames(21, 16, 3, fading='full')
    12
    >>> stft(np.zeros(21), 16, 3).shape
    (12, 9)
    >>> _samples_to_stft_frames(21, 16, 3)
    3
    >>> stft(np.zeros(21), 16, 3, fading=None).shape
    (3, 9)
    """

    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [None, False]:
        pad_width = (size - shift)
        samples = samples + (1 + (fading != 'half')) * pad_width

    # I changed this from np.ceil to math.ceil, to yield an integer result.
    frames = (samples - size + shift) / shift
    if pad:
        return ceil(frames)
    return int(frames)


def _stft_frames_to_samples(
        frames, size, shift, fading=None
):
    """
    Calculates samples in time domain from STFT frames
    :param frames: Number of STFT frames.
    :param size: window_length often equal to FFT size.
                 The name size should be marked as deprecated and replaced with
                 window_length.
    :param shift: Hop in samples.
    :return: Number of samples in time domain.

    >>> _stft_frames_to_samples(2, 16, 4)
    20
    """
    samples = frames * shift + size - shift

    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [None, False]:
        pad_width = (size - shift)
        samples -= (1 + (fading != 'half')) * pad_width
    return samples


def sample_index_to_stft_frame_index(sample, window_length, shift, fading='full'):
    """
    Calculates the best frame index for a given sample index

    :param sample: Sample index in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Best STFT frame index.


    ## ## ## ##
       ## ## ## ##
          ## ## ## ##
             ## ## ## ##
    00 00 01 12 23 34 45


    ## ## ## ##
     # ## ## ## #
       ## ## ## ##
        # ## ## ## #
    00 00 01 23 5 ...
          12 34 6 ...

    ## ## ## #
       ## ## ## #
          ## ## ## #
             ## ## ## #
    ## ## ## #
    00 00 01 12 23 34 45

    >>> [sample_index_to_stft_frame_index(i, 8, 1, fading=None) for i in range(12)]
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    >>> [sample_index_to_stft_frame_index(i, 8, 2, fading=None) for i in range(10)]
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    >>> [sample_index_to_stft_frame_index(i, 7, 2, fading=None) for i in range(10)]
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    >>> [sample_index_to_stft_frame_index(i, 7, 1, fading=None) for i in range(10)]
    [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]

    >>> [sample_index_to_stft_frame_index(i, 8, 1, fading='full') for i in range(12)]
    [7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    >>> [sample_index_to_stft_frame_index(i, 8, 2, fading='full') for i in range(10)]
    [3, 3, 3, 3, 4, 4, 5, 5, 6, 6]
    >>> [sample_index_to_stft_frame_index(i, 7, 2, fading='full') for i in range(10)]
    [3, 3, 3, 3, 4, 4, 5, 5, 6, 6]
    >>> [sample_index_to_stft_frame_index(i, 7, 1, fading='full') for i in range(10)]
    [6, 6, 6, 6, 7, 8, 9, 10, 11, 12]

    >>> stft(np.zeros([8]), size=8, shift=2).shape
    (7, 5)
    >>> stft(np.zeros([8]), size=8, shift=1).shape
    (15, 5)
    >>> stft(np.zeros([8]), size=8, shift=4).shape
    (3, 5)
    """

    if (window_length + 1) // 2 > sample:
        frame = 0
    else:
        frame = (sample - (window_length + 1) // 2) // shift + 1

    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [None, False]:
        pad_width = (window_length - shift)
        if fading == 'half':
            pad_width //= 2
        frame = frame + ceil(pad_width / shift)

    return frame


def _biorthogonal_window_loopy(analysis_window, shift):
    """
    This version of the synthesis calculation is as close as possible to the
    Matlab implementation in terms of variable names.

    The results are equal.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        for sample_index in range(0, number_of_shifts+1):
            analysis_index = synthesis_index + sample_index * shift

            if analysis_index + 1 < fft_size:
                sum_of_squares[synthesis_index] \
                    += analysis_window[analysis_index] ** 2

    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size

    # Why? Line created by Hai, Lukas does not know, why it exists.
    synthesis_window *= fft_size

    return synthesis_window


def _biorthogonal_window(analysis_window, shift):
    """
    This is a vectorized implementation of the window calculation. It is much
    slower than the variant using for loops.

    The implementation follows equation A.92 in
    Krueger, A. Modellbasierte Merkmalsverbesserung zur robusten automatischen
    Spracherkennung in Gegenwart von Nachhall und Hintergrundstoerungen
    Paderborn, Universitaet Paderborn, Diss., 2011, 2011
    """
    fft_size = len(analysis_window)
    assert np.mod(fft_size, shift) == 0
    number_of_shifts = len(analysis_window) // shift

    sum_of_squares = np.zeros(shift)
    for synthesis_index in range(0, shift):
        sample_index = np.arange(0, number_of_shifts+1)
        analysis_index = synthesis_index + sample_index * shift
        analysis_index = analysis_index[analysis_index + 1 < fft_size]
        sum_of_squares[synthesis_index] \
            = np.sum(analysis_window[analysis_index] ** 2)
    sum_of_squares = np.kron(np.ones(number_of_shifts), sum_of_squares)
    synthesis_window = analysis_window / sum_of_squares / fft_size

    # Why? Line created by Hai, Lukas does not know, why it exists.
    synthesis_window *= fft_size

    return synthesis_window


def _biorthogonal_window_brute_force(analysis_window, shift,
                                     use_amplitude=False):
    """
    The biorthogonal window (synthesis_window) must verify the criterion:
        synthesis_window * analysis_window plus it's shifts must be one.
        1 == sum m from -inf to inf over (synthesis_window(n - mB) * analysis_window(n - mB))
        B ... shift
        n ... time index
        m ... shift index

    :param analysis_window:
    :param shift:
    :return:

    >>> analysis_window = signal.windows.blackman(4+1)[:-1]
    >>> print(analysis_window)
    [-1.38777878e-17  3.40000000e-01  1.00000000e+00  3.40000000e-01]
    >>> synthesis_window = _biorthogonal_window_brute_force(analysis_window, 1)
    >>> print(synthesis_window)
    [-1.12717575e-17  2.76153346e-01  8.12215724e-01  2.76153346e-01]
    >>> mult = analysis_window * synthesis_window
    >>> sum(mult)
    1.0000000000000002
    """
    size = len(analysis_window)

    influence_width = (size - 1) // shift

    denominator = np.zeros_like(analysis_window)

    if use_amplitude:
        analysis_window_square = analysis_window
    else:
        analysis_window_square = analysis_window ** 2
    for i in range(-influence_width, influence_width + 1):
        denominator += roll_zeropad(analysis_window_square, shift * i)

    if use_amplitude:
        synthesis_window = 1 / denominator
    else:
        synthesis_window = analysis_window / denominator
    return synthesis_window


_biorthogonal_window_fastest = _biorthogonal_window_brute_force


def istft(
        stft_signal,
        size: int=1024,
        shift: int=256,
        *,
        window: [str, typing.Callable]=signal.windows.blackman,
        fading: typing.Optional[typing.Union[bool, str]] = 'full',
        window_length: int=None,
        symmetric_window: bool=False,
        num_samples: int=None,
        pad: bool=True,
):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    ..note::
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to
        the unmodified! analysis window.

    :param stft_signal: Single channel complex STFT signal
        with dimensions (..., frames, size/2+1).
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Removes the additional padding, if done during STFT.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :param symmetric_window: symmetric or periodic window. Assume window is
        periodic. Since the implementation of the windows in scipy.signal have a
        curious behaviour for odd window_length. Use window(len+1)[:-1]. Since
        is equal to the behaviour of MATLAB.
    :param num_samples: None or the number of samples that the original time
        signal has. When given, check, that the backt transformed signal
        has a valid number of samples and shorten the signal to the original
        length. (Does only work when pad is True).
    :param pad: Necessary when num_samples is not None. This arguments is only
        for the forward transform nessesary and not for the inverse.
        Here it is used, to check that num_samples is valid.

    :return: Single channel complex STFT signal
    :return: Single channel time signal.
    """
    # Note: frame_axis and frequency_axis would make this function much more
    #       complicated
    stft_signal = np.array(stft_signal)

    assert stft_signal.shape[-1] == size // 2 + 1, str(stft_signal.shape)

    if window_length is None:
        window_length = size

    window = _get_window(
        window=window,
        symmetric_window=symmetric_window,
        window_length=window_length,
    )

    window = _biorthogonal_window_fastest(window, shift)

    # window = _biorthogonal_window_fastest(
    #     window, shift, use_amplitude_for_biorthogonal_window)
    # if disable_sythesis_window:
    #     window = np.ones_like(window)

    time_signal = np.zeros(
        (*stft_signal.shape[:-2],
         stft_signal.shape[-2] * shift + window_length - shift))

    # Get the correct view to time_signal
    time_signal_seg = segment_axis(
        time_signal, window_length, shift, end=None
    )

    # Unbuffered inplace add
    np.add.at(
        time_signal_seg,
        ...,
        window * np.real(irfft(stft_signal))[..., :window_length]
    )
    # The [..., :window_length] is the inverse of the window padding in rfft.

    # Compensate fade-in and fade-out

    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [None, False]:
        pad_width = (window_length - shift)
        if fading == 'half':
            pad_width /= 2
        time_signal = time_signal[
            ..., int(pad_width):time_signal.shape[-1] - ceil(pad_width)]

    if num_samples is not None:
        if pad:
            assert time_signal.shape[-1] >= num_samples, (time_signal.shape, num_samples)
            assert time_signal.shape[-1] < num_samples + shift, (time_signal.shape, num_samples)
            time_signal = time_signal[..., :num_samples]
        else:
            raise ValueError(
                pad,
                'When padding is False in the stft, the signal is cutted.'
                'This operation can not be inverted.',
            )

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    :param stft_signal: Complex STFT signal with dimensions
        #time_frames times #frequency_bins.
    :return: Real spectrogram with same dimensions as input.

    Note: Special version of paderbox.math.scalar.abs_square
    """

    spectrogram = stft_signal.real**2 + stft_signal.imag**2
    return spectrogram


def spectrogram(time_signal, *args, **kwargs):
    """ Thin wrapper of stft with power spectrum calculation.

    :param time_signal:
    :param args:
    :param kwargs:
    :return:
    """
    return stft_to_spectrogram(stft(time_signal, *args, **kwargs))


def spectrogram_to_energy_per_frame(spectrogram):
    """
    The energy per frame is sometimes used as an additional feature to the MFCC
    features. Here, it is calculated from the power spectrum.

    :param spectrogram: Real valued power spectrum.
    :return: Real valued energy per frame.
    """
    energy = np.sum(spectrogram, 1)

    # If energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    return energy


def get_stft_center_frequencies(size=1024, sample_rate=16000):
    """
    It is often necessary to know, which center frequency is
    represented by each frequency bin index.

    :param size: Scalar FFT-size.
    :param sample_rate: Scalar sample frequency in Hertz.
    :return: Array of all relevant center frequencies
    """
    frequency_index = np.arange(0, size/2 + 1)
    return frequency_index * sample_rate / size


class STFT:
    def __init__(
            self,
            shift: int,
            size: int,
            window_length: int = None,
            window: str = "blackman",
            symmetric_window: bool = False,
            pad: bool = True,
            fading: typing.Optional[typing.Union[bool, str]] = 'full'
    ):
        """
        Transforms audio data to STFT.
        Also allows to invert stft as well as reconstruct phase information
        from magnitudes using griffin lim algorithm.

        Args:
            shift:
            size:
            window_length:
            window:
            symmetric_window:
            fading:
            pad:

        >>> stft = STFT(160, 512, fading='full')
        >>> audio_data=np.zeros(8000)
        >>> x = stft(audio_data)
        >>> x.shape
        (53, 257)
        """
        self.shift = shift
        self.size = size
        self.window_length = window_length if window_length is not None \
            else size
        if isinstance(window, str):
            window = getattr(signal.windows, window)
        self.window = window
        self.symmetric_window = symmetric_window
        self.fading = fading
        self.pad = pad

    def __call__(self, x):
        """
        Performs stft

        Args:
            x: time signal

        Returns:

        """
        x = stft(
            x,
            size=self.size,
            shift=self.shift,
            window_length=self.window_length,
            window=self.window,
            symmetric_window=self.symmetric_window,
            axis=-1,
            fading=self.fading,
            pad=self.pad
        )  # (..., T, F)

        return x

    def inverse(self, x):
        """
        Computes inverse stft

        Args:
            x: stft

        Returns:

        """
        #  x: (C, T, F)
        return istft(
            x,
            size=self.size,
            shift=self.shift,
            window_length=self.window_length,
            window=self.window,
            symmetric_window=self.symmetric_window,
            fading=self.fading
        )

    def samples_to_frames(self, samples):
        """
        Calculates number of STFT frames from number of samples in time domain.

        Args:
            samples: Number of samples in time domain.

        Returns:
            Number of STFT frames.

        """
        return _samples_to_stft_frames(
            samples, self.window_length, self.shift,
            pad=self.pad, fading=self.fading
        )

    def sample_index_to_frame_index(self, sample_index):
        """
        Calculates the best frame index for a given sample index

        Args:
            sample_index:

        Returns:

        """
        return sample_index_to_stft_frame_index(
            sample_index, self.window_length, self.shift, fading=self.fading
        )

    def frames_to_samples(self, frames):
        """
        Calculates samples in time domain from STFT frames

        Args:
            frames: number of frames in STFT

        Returns: number of samples in time signal

        """
        return _stft_frames_to_samples(
            frames, self.window_length, self.shift, fading=self.fading
        )
