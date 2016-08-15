import numpy as np
from nt.transform import get_stft_center_frequencies
import math
from nt.utils.misc import interleave
from scipy.special import hyp1f1
from scipy.interpolate import interp1d
from random import shuffle
from nt.utils import Timer
from nt.utils import json_utils
from collections import defaultdict
from nt.nn.data_fetchers import JsonCallbackFetcher
from nt.io.audioread import audioread

from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from nt.speech_enhancement.beamformer import get_pca


def _is_power_of_two(number):
    """
    >>> _is_power_of_two(0)
    False
    >>> _is_power_of_two(1)
    True
    >>> _is_power_of_two(2)
    True
    """
    return not (number == 0) and not number & (number - 1)


def normalize_observation(
        signal, unit_norm=True, phase_norm=False, frequency_norm=False,
        max_sensor_distance=None, shrink_factor=1.2,
        fft_size=1024, sample_rate=16000, sound_velocity=343
):
    """ Different feature normalizations.

    Args:
        signal: STFT signal with shape (..., sensors, frequency, time).
        phase_norm: The first sensor element will be real and positive.
        unit_norm: Normalize vector length to length one.
        frequency_norm:
        max_sensor_distance:
        shrink_factor:
        fft_size:
        sample_rate:
        sound_velocity:

    Returns:

    """
    D, F, T = signal.shape[-3:]
    assert _is_power_of_two(F - 1)

    signal = np.copy(signal)

    if unit_norm:
        signal /= np.linalg.norm(signal, axis=-3, keepdims=True)

    if phase_norm:
        signal *= np.exp(-1j * np.angle(signal[..., 0, :, :]))

    if frequency_norm:
        frequency = get_stft_center_frequencies(fft_size, sample_rate)
        assert len(frequency) == F
        norm_factor = sound_velocity / (
            2 * frequency * shrink_factor * max_sensor_distance
        )
        norm_factor = np.nan_to_num(norm_factor)
        if norm_factor[-1] < 1:
            raise ValueError(
                'Distance between the sensors too high: {:.2} > {:.2}'.format(
                    max_sensor_distance, sound_velocity / (2 * frequency[-1])
                )
            )

        # Add empty dimensions at start and end
        norm_factor = norm_factor.reshape((signal.ndim - 2) * (1,) + (-1, 1,))

        signal = np.abs(signal) * np.exp(1j * np.angle(signal) * norm_factor)

    return signal


class ComplexWatson:
    """
    >>> from nt.speech_enhancement import bss
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> scales = [
    ...     np.arange(0, 0.01, 0.001),
    ...     np.arange(0, 20, 0.01),
    ...     np.arange(0, 100, 1)
    ... ]
    >>> functions = [
    ...     bss.ComplexWatson.log_norm_low_concentration,
    ...     bss.ComplexWatson.log_norm_medium_concentration,
    ...     bss.ComplexWatson.log_norm_high_concentration
    ... ]
    >>>
    >>> f, axis = plt.subplots(1, 3)
    >>> for ax, scale in zip(axis, scales):
    ...     result = [fn(scale, 6) for fn in functions]
    ...     [ax.plot(scale, np.log(r), '--') for r in result]
    ...     ax.legend(['low', 'middle', 'high'])
    >>> plt.show()
    """

    @staticmethod
    def pdf(x, loc, scale):
        """ Calculates pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        return np.exp(ComplexWatson.log_pdf(x, loc, scale))

    @staticmethod
    def log_pdf(x, loc, scale):
        """ Calculates logarithm of pdf function.

        Args:
            x: Assumes shape (..., D).
            loc: Mode vector. Assumes corresponding shape (..., D).
            scale: Concentration parameter with shape (...).

        Returns:
        """
        # For now, we assume that the caller does proper expansion
        assert x.ndim == loc.ndim
        assert x.ndim - 1 == scale.ndim

        result = np.einsum('...d,...d', x, loc.conj())
        result = result.real ** 2 + result.imag ** 2
        result *= scale
        result -= ComplexWatson.log_norm(scale, x.shape[-1])
        return result

    @staticmethod
    def log_norm_low_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Good at very low concentrations but starts to drop of at 20.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        # Mardia1999Watson Equation (4), Taylor series
        b_range = range(dimension, dimension + 20 - 1 + 1)
        b_range = np.asarray(b_range)[None, :]

        return (
            np.log(2) + dimension * np.log(np.pi) -
            np.log(math.factorial(dimension - 1)) +
            np.log(1 + np.sum(np.cumprod(scale[:, None] / b_range, -1), -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_medium_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        Almost complete range of interest and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.flatten()

        # Function is unstable at zero. Scale needs to be float for this to work
        scale[scale < 1e-2] = 1e-2

        r_range = range(dimension - 2 + 1)
        r = np.asarray(r_range)[None, :]

        # Mardia1999Watson Equation (3)
        temp = scale[:, None] ** r * np.exp(-scale[:, None]) / \
               np.asarray([math.factorial(_r) for _r in r_range])

        return (
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale +
            np.log(1. - np.sum(temp, -1))
        ).reshape(shape)

    @staticmethod
    def log_norm_high_concentration(scale, dimension):
        """ Calculates logarithm of pdf function.
        High concentration above 10 and dimension below 8.
        """
        scale = np.asfarray(scale)
        shape = scale.shape
        scale = scale.ravel()

        return (
            np.log(2.) + dimension * np.log(np.pi) +
            (1. - dimension) * np.log(scale) + scale
        ).reshape(shape)

    log_norm = log_norm_medium_concentration


def frequency_permutation_alignment(
        mask,
        segment_start=100,
        segment_width=100,
        segment_shift=20,
        main_iterations=20,
        sub_iterations=2
):
    """

    Args:
        mask: Shape (K, F, T)
        segment_start:
        segment_width:
        segment_shift:
        main_iterations:
        sub_iterations:

    Returns:

    """
    K, F, T = mask.shape
    alignment_plan_lower_start = list(
        range(segment_start + segment_shift, F - segment_width, segment_shift))

    alignment_plan_higher_start = list(
        range(segment_start - segment_shift, 0, -segment_shift))

    alignment_plan_start = list(
        interleave(alignment_plan_lower_start, alignment_plan_higher_start))

    alignment_plan = [
                         [main_iterations, segment_start,
                          segment_start + segment_width]
                     ] + [[sub_iterations, s, s + segment_width] for s in
                          alignment_plan_start]

    alignment_plan[2 * len(alignment_plan_higher_start)][1] = 0
    alignment_plan[-1][2] = F

    # (K, F, T)
    features = mask / np.linalg.norm(mask, axis=-1, keepdims=True)

    # TODO: Write without copy.
    mapping = np.copy(np.broadcast_to(np.arange(K)[:, None], (K, F)))

    for iterations, start, end in alignment_plan:
        for _ in range(iterations):
            # (K, T)
            centroid = np.sum(features[:, start:end, :], axis=1)
            centroid /= np.linalg.norm(centroid, axis=-1, keepdims=True)

            break_flag = False
            for f in range(start, end):
                reverse_permutation = _align_segment(centroid,
                                                     features[:, f, :])
                if not (reverse_permutation == list(range(K))).all():
                    break_flag = True
                    features[:, f, :] = features[reverse_permutation, f, :]
                    mapping[:, f] = mapping[reverse_permutation, f]
            if break_flag:
                break

    # corrected_mask = np.zeros_like(mask)
    # for f in range(F):
    #     corrected_mask[:, f, :] = mask[mapping[:, f], f]

    return mapping


def _align_segment(prototype, permuted):
    # Takes a lot of time:
    # np.testing.assert_equal(prototype.shape, permuted.shape)
    K = prototype.shape[0]

    c_matrix = np.dot(prototype, permuted.T)
    # Takes a lot of time:
    # np.testing.assert_equal(c_matrix.shape, (K, K))

    reverse_permutation = np.zeros((K,), dtype=np.int)
    estimated_permutation = np.zeros((K,), dtype=np.int)

    for k in range(K):
        # TODO: Can be written with just one max call
        c_max, index_0 = np.max(c_matrix, axis=0), np.argmax(c_matrix, axis=0)

        index_1 = np.argmax(c_max)
        c_matrix[index_0[index_1], :] = -1
        c_matrix[:, index_1] = -1
        reverse_permutation[index_0[index_1]] = index_1
        estimated_permutation[index_1] = index_0[index_1]

    return reverse_permutation


def source_permutation_alignment(mask_estimate, target):
    """ This solves the global permutation problem, if masks are not too wild.

    Args:
        mask_estimate: Assumes shape (K, F, T)
        target: Assumes shape (K, F, T)

    Returns: Mapping, i.e. [2, 0, 1] to map mask_estimate.

    """
    K = mask_estimate.shape[0]

    def similarity(a, b):
        return np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b)

    similarity_matrix = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            similarity_matrix[k, l] = similarity(mask_estimate[k], target[l])

    return similarity_matrix_to_mapping(similarity_matrix)


def similarity_matrix_to_mapping(similarity_matrix):
    """ Calculate mapping from similarity matrix.

    Args:
        similarity_matrix: Similarity matrix between ``a`` and ``b``.
            Row index: Index of observations ``a``.
            Column index: Index of observations ``b``.

    Returns: Mapping from ``a`` to ``b``. That means, apply the mapping on
        variables corresponding to ``a`` will result in ``a`` being aligned
        to ``b``.

    >>> similarity_matrix = np.asarray([[1, 1, 0], [0, 1, 1], [1, 2, 1]])
    >>> similarity_matrix_to_mapping(similarity_matrix)
    [0, 2, 1]

    >>> similarity_matrix = np.asarray([[0, 1, 1], [1, 1, 0], [1, 2, 1]])
    >>> similarity_matrix_to_mapping(similarity_matrix)
    [1, 2, 0]

    """
    K = similarity_matrix.shape[0]
    mapping = [-1] * K

    for _ in range(K):
        a, b = np.unravel_index(np.argmax(similarity_matrix), (K, K))
        similarity_matrix[a, :] = 0
        similarity_matrix[:, b] = 0
        mapping[b] = a

    return mapping


def apply_alignment_inplace(*list_of_statistics, mapping):
    K, F = mapping.shape

    for statistics in list_of_statistics:
        for f in range(F):
            statistics[:K, f] = statistics[mapping[:, f], f]


class HypergeometricRatioSolver:
    """ This is twice as slow as interpolation with Tran Vu's C-code, but works.

    >>> a = np.logspace(-3, 2.5, 100)
    >>> hypergeometric_ratio_inverse = HypergeometricRatioSolver()
    >>> hypergeometric_ratio_inverse(a, 3)
    """

    # TODO: Is it even necessary to cache this?
    # TODO: Possibly reduce number of markers, if speedup is necessary at all.
    # TODO: Improve handling for very high and very low values.

    def __init__(self, max_concentration=100):
        x = np.logspace(-3, np.log10(max_concentration), 100)
        self.list_of_splines = 11 * [None]
        for d in range(2, 11):
            y = hyp1f1(2, d + 1, x) / (d * hyp1f1(1, d, x))
            self.list_of_splines[d] = interp1d(
                y, x, kind='quadratic',
                assume_sorted=True,
                bounds_error=False,
                fill_value=(0, max_concentration)
            )

    def __call__(self, a, D):
        return self.list_of_splines[D](a)


class Fetcher(JsonCallbackFetcher):
    def _get_utterance_list(self):
        speaker_a = list(self.flist.keys())
        speaker_b = list(self.flist.keys())
        shuffle(speaker_a)
        shuffle(speaker_b)
        return [(a, b) for a, b in zip(speaker_a, speaker_b) if not a == b]

    def _read_utterance(self, utt_list):
        data_dict = defaultdict(list)
        with Timer() as t:
            for utt in utt_list:
                for channel in sorted(self.feature_channels):
                    try:
                        wav_file = json_utils.get_channel_for_utt(
                            self.flist, channel, utt
                        )
                    except KeyError as e:
                        if not self.ignore_unavailable_channels:
                            raise e
                    else:
                        data = audioread(wav_file, sample_rate=self.sample_rate)
                        data = np.atleast_2d(data)
                        ch_group = channel.split('/')[0]
                        data_dict[ch_group].append([utt, data])

        self.io_time.value += t.secs

        assert len(data_dict) > 0, 'Did not load any audio data.'
        return data_dict


def get_list_of_signal_variances(list_of_signals, list_of_signal_samples):
    def _var(x, duration):
        assert np.isrealobj(x)
        assert x.ndim in [1, 2]
        channels = 1 if x.ndim == 1 else x.shape[0]
        return np.sum(x ** 2) / duration / channels

    return [_var(x, T) for x, T in zip(list_of_signals, list_of_signal_samples)]


def _db(a):
    return 10 * np.log10(a)


def get_multi_speaker_sxr(
        list_of_source_signals,
        list_of_source_signal_samples,
        noise_signal,
        noise_signal_samples
):
    """ Calculate multi speaker and multi channel sxr.

    This may change to take input similar to mask estimation functions.

    Args:
        list_of_source_signals: Each shape is assumed to be (D, T_k) or (T_k,)
        list_of_source_signal_samples: Each duration [T_1, ..., T_K]
        noise_signal: Assumed to have shape (D, T_N) or (T_N,)
        noise_signal_samples: Scalar T_N
    """
    assert len(list_of_source_signals) == len(list_of_source_signal_samples)

    list_of_source_variances = get_list_of_signal_variances(
        list_of_source_signals, list_of_source_signal_samples
    )
    noise_variance = get_list_of_signal_variances(
        [noise_signal], [noise_signal_samples]
    )[0]

    snr = np.mean(list_of_source_variances) / noise_variance
    single_speaker_snr = [a / noise_variance for a in list_of_source_variances]

    return _db(snr), _db(single_speaker_snr)


def em(
        Y, mixture_components=3, iterations=100,
        affiliations=None, alignment=True
):
    Y_normalized = normalize_observation(Y, frequency_norm=False)
    Y_normalized_for_psd = np.copy(Y_normalized[0], 'C')
    Y_normalized_for_pdf = np.copy(Y_normalized.transpose(0, 2, 3, 1), 'C')

    if affiliations is None:
        affiliations = np.random.dirichlet(
            mixture_components * [1 / mixture_components], size=(Y.shape[-2:])
        ).transpose((2, 0, 1))
    hypergeometric_ratio_inverse = HypergeometricRatioSolver()

    for i in range(iterations):
        pi = affiliations.mean(axis=-1)
        Phi = get_power_spectral_density_matrix(
            Y_normalized_for_psd,
            np.copy(affiliations, 'C'),
            sensor_dim=0, source_dim=0,
            time_dim=-1
        )
        W, eigenvalues = get_pca(Phi)
        kappa = hypergeometric_ratio_inverse(eigenvalues, W.shape[-1])

        affiliations = pi[..., None] * ComplexWatson.pdf(
            Y_normalized_for_pdf,
            np.copy(W[:, :, None, :], 'C'),
            kappa[:, :, None]
        )
        affiliations /= np.sum(affiliations, axis=0, keepdims=True)

    if alignment:
        mapping = frequency_permutation_alignment(affiliations)
        apply_alignment_inplace(affiliations, pi, W, kappa, mapping=mapping)

    return dict(
        affiliations=affiliations,
        kappa=kappa,
        W=W,
        pi=pi
    )
