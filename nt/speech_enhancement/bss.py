from collections import defaultdict
from random import shuffle

import nt
import numpy as np
from nt.io.audioread import audioread
from nt.nn.data_fetchers import JsonCallbackFetcher
from nt.speech_enhancement.beamformer import get_power_spectral_density_matrix
from nt.utils import Timer
from nt.utils import json_utils
from nt.utils.misc import interleave


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

    Main difference to nt/evaluation/sxr.py is use of signal_samples.

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


def set_snr(x, n, snr):
    """Rescale sources and noise signal with given single speaker snr."""
    x /= np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, keepdims=True))
    nt.speech_enhancement.noise.set_snr(x, n, snr)
