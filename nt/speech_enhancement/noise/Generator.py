from abc import ABCMeta, abstractmethod
from collections import Iterable
import numpy as np
from nt.speech_enhancement.noise.utils import set_snr
from scipy.signal import lfilter
from nt.speech_enhancement.noise.spherical_habets import _sinf_3D_py
import nt.testing as tc
import nt.io.audioread as ar
from nt.database.noisex92 import helper
import json
from nt.io.audioread import audioread


class NoiseGeneratorTemplate:
    """
    Make sure, your implementation of get_noise() uses the provided random
    number generator.
    Setting the random number generator state globally is discouraged.
    """
    __metaclass__ = ABCMeta

    def get_noise_for_signal(self, time_signal, *, snr, seed=None,
                             rng_state=None, **kwargs):
        """
        Args:
            time_signal:
            snr: SNR or single speaker SNR.
            seed: Seed used to create a new random number generator object.
            **kwargs:
        """
        if rng_state is None:
            rng_state = np.random if seed is None \
                else np.random.RandomState(seed)
        noise_signal = self.get_noise(
            *time_signal.shape,
            rng_state=self.rng_state,
            **kwargs
        )
        set_snr(time_signal, noise_signal, snr)
        return noise_signal

    def get_noise(self, *shape, seed=None, rng_state=None, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if rng_state is None:
            rng_state = np.random if seed is None else np.random.RandomState(seed)
        return self._get_noise(shape, rng_state, **kwargs)

    @abstractmethod
    def _get_noise(self, shape, rng_state=None, **kwargs):
        pass


class NoiseGeneratorWhite(NoiseGeneratorTemplate):
    """
    Example:

    >>> from nt.evaluation.sxr import input_sxr
    >>> time_signal = np.random.normal(size=(1000,))
    >>> ng = NoiseGeneratorWhite()
    >>> n = ng.get_noise_for_signal(time_signal, 20)
    >>> SDR, SIR, SNR = input_sxr(time_signal[:, None, None], n[:, None, None])
    >>> round(SNR, 1)
    20.0

    >>> from nt.evaluation.sxr import input_sxr
    >>> time_signal = np.random.normal(size=(1000, 2))
    >>> ng = NoiseGeneratorWhite()
    >>> n = ng.get_noise_for_signal(time_signal, 20)
    >>> n.shape
    (1000, 2)

    >>> time_signal.shape
    (1000, 2)

    >>> SDR, SIR, SNR = input_sxr(time_signal[:, :, None], n)
    >>> round(SNR, 1)
    20.0
    """
    def _get_noise(self, shape, rng_state, **kwargs):
        noise_signal = rng_state.normal(size=shape)
        return noise_signal


class NoiseGeneratorChimeBackground(NoiseGeneratorTemplate):
    """ Generate random background noise by sampling from Chime background.

    Shape of your input signal is assumed to be (channels, samples).
    You can create the file by running this code:
    ``python -m nt.database.chime.create_background_json``

    >>> ng = NoiseGeneratorChimeBackground('chime_bss.json', flist='train')
    >>> noise = ng.get_noise((3, 16000), np.random)
    >>> print(noise.shape)
    (3, 16000)
    """
    def __init__(
            self, json_src, flist=None, sampling_rate=16000, max_channels=6
    ):
        flist = 'all' if flist is None else flist
        with open(json_src) as f:
            database = json.load(f)

        self.sampling_rate = sampling_rate
        self.max_channels = max_channels
        self.flist = database[flist]
        self.utterance_list = sorted(self.flist['wav'].keys())

    def get_noise(self, shape, rng_state, **kwargs):
        D, T = shape[-2:]

        channels = rng_state.choice(self.max_channels, D, replace=False)
        utt_id = rng_state.randint(len(self.flist))
        utt_id = self.utterance_list[utt_id]
        max_samples = self.flist['annotations'][utt_id]['samples']
        start = rng_state.randint(max_samples - T)

        noise_list = []
        for channel in channels:
            noise_list.append(audioread(
                self.flist['wav'][utt_id]['CH{}'.format(channel + 1)],
                offset=start/self.sampling_rate, duration=T/self.sampling_rate
            ))

        return np.stack(noise_list)


class NoiseGeneratorPink(NoiseGeneratorTemplate):
    """
    See example code of ``NoiseGeneratorWhite``.
    """
    def get_noise(self, shape, rng_state, **kwargs):
        """Generates pink noise. You still need to rescale it to your needs.

        This code was taken from the book
        Spectral Audio Signal Processing, by Julius O. Smith III,
        W3K Publishing, 2011, ISBN 978-0-9745607-3-1:
        `Spectral Audio Signal Processing
        <https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html>`

        Alternative implementation can be found in
        `dsp.stackexchange.com <http://dsp.stackexchange.com/a/376/8515>`.

        :param shape: Shape of desired noise. Time is assumed to be last axis.
        :param rng_state:
        """

        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, - 2.494956002, 2.017265875, -0.522189400]

        # Original Matlab Code:
        #     transient_samples = round(log(1000)/(1-max(abs(roots(A)))));
        transient_samples = 1430

        samples = shape[-1]
        size = (shape[:-1] + (transient_samples + samples,))
        v = rng_state.normal(size=size)

        # Apply 1/F roll-off to PSD
        noise = lfilter(b, a, v, axis=-1)

        # Skip transient response
        noise = noise[..., transient_samples:]
        return noise


#class NoiseGeneratorNoisex92(NoiseGeneratorTemplate):
#     def __init__(self, label=None, sample_rate=16000):
#
#         self.labels = helper.get_labels()
#         if label is not None:
#             if label not in self.labels:
#                 raise KeyError('The label "{label}" does not exist. '
#                                'Please choose a valid label from following list: '
#                                '{l}'.format(label=label, l=', '.join(self.labels)))
#             self.labels = [label]
#         self.audio_datas = list()
#         for l in self.labels:
#                 path = helper.get_path_for_label(l, sample_rate)
#                 self.audio_datas += [ar.audioread(path, sample_rate=sample_rate)]
#         self.sample_rate = sample_rate
#
#     @_decorator_noise_generator_set_snr
#     def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
#         """
#
#         Example:
#
#         >>> import nt.evaluation.sxr as sxr
#         >>> time_signal = numpy.random.randn(16000)
#         >>> n_gen = NoiseGeneratorNoisex92(sample_rate = 16000)
#         >>> n = n_gen.get_noise_for_signal(time_signal, 20, seed=1)
#         >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
#         >>> SNR
#         20
#         >>> n = n_gen.get_noise_for_signal(time_signal, 20)
#         >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
#         >>> SNR
#         20
#         >>> label = 'destroyereng'
#         >>> n_gen = NoiseGeneratorNoisex92(label, sample_rate = 16000)
#         >>> label = 'destroyerengine'
#         >>> n_gen = NoiseGeneratorNoisex92(label, sample_rate = 16000)
#         >>> n = n_gen.get_noise_for_signal(time_signal, 20)
#         >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
#         >>> SNR
#         20
#         """
#         idx = numpy.random.choice(len(self.audio_datas))
#         audio_data = self.audio_datas[idx]
#         # self.last_label = self.labels[idx]
#         if time_signal.shape[0] <= audio_data.shape[0]:
#             seq = numpy.random.randint(0, audio_data.shape[0] - time_signal.shape[0])
#         else:
#             raise ValueError('Length of Time Signal is longer then the length of a possible noisex92 Noise Signal!')
#         noise_signal = audio_data[seq: seq + time_signal.shape[0]]
#         return noise_signal


# class NoiseGeneratorSpherical(NoiseGeneratorTemplate):
#
#     def __init__(self, sensor_positions, *, sample_axis=-2, channel_axis=-1, sample_rate=16000, c=340,
#                  number_of_cylindrical_angels=256):
#
#         assert sensor_positions.shape[0] == 3
#
#         self.sample_axis = sample_axis
#         self.channel_axis = channel_axis
#
#         self.sensor_positions = sensor_positions
#         _, self.number_of_channels = sensor_positions.shape
#
#         self.sample_rate = sample_rate
#         self.sound_velocity = c
#         self.number_of_cylindrical_angels = number_of_cylindrical_angels
#
#     @_decorator_noise_generator_set_snr
#     def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
#         """
#         Example:
#
#         >>> import nt.evaluation.sxr as sxr
#         >>> from nt.utils.math_ops import sph2cart
#         >>> time_signal = numpy.random.randn(1000, 3)
#         >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
#         >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
#         >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
#         >>> n_gen = NoiseGeneratorSpherical(P)
#         >>> n = n_gen.get_noise_for_signal(time_signal, 20)
#         >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n[:, :, None])
#         >>> SNR
#         20.0
#
#         >>> import nt.evaluation.sxr as sxr
#         >>> from nt.utils.math_ops import sph2cart
#         >>> time_signal = numpy.random.randn(1000, 3)
#         >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
#         >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
#         >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
#         >>> n_gen = NoiseGeneratorSpherical(P)
#         >>> n = n_gen.get_noise_for_signal(time_signal, 20)
#         >>> n.shape
#         (1000, 3)
#         >>> time_signal.shape
#         (1000, 3)
#         >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, numpy.newaxis], n)
#         >>> SNR
#         20.0
#
#         """
#         tc.assert_equal(time_signal.shape[self.channel_axis], self.number_of_channels)
#
#         # shape = time_signal.shape
#         noise_signal = _sinf_3D_py(self.sensor_positions, time_signal.shape[self.sample_axis])
#         return noise_signal
#         # return self._normalize_noise(time_signal,snr,noise_signal)
#
#
# class NoiseGeneratorMix:
#     """
#
#     Example:
#
#     >>> import nt.evaluation.sxr as sxr
#     >>> time_signal = numpy.random.randn(1000)
#     >>> n_gens = [NoiseGeneratorWhite(), NoiseGeneratorPink()]
#     >>> n_gen = NoiseGeneratorMix(n_gens, max_different_types=2)
#     >>> n = n_gen.get_noise_for_signal(time_signal, 20)
#     >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
#     >>> SNR
#     20.0
#
#
#     """
#
#     def __init__(self, noise_generators, max_different_types=1, probabilities=None):
#         self.noise_generators = noise_generators
#         for n in noise_generators:
#             assert isinstance(n, NoiseGeneratorTemplate)
#         self.max_different_types = max_different_types
#         if probabilities:
#             self.probabilities = probabilities / sum(probabilities)
#         else:
#             self.probabilities = numpy.ones(len(noise_generators)) * 1 / len(noise_generators)
#
#     def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
#
#         numpy.random.seed(seed=seed)
#         # replace = True ; Ziehen mit zurücklegen
#         # replace = False ; Ziehen ohne zurücklegen
#         noise_idx = numpy.random.choice(len(self.noise_generators), self.max_different_types, replace=True,
#                                         p=self.probabilities)
#
#         noise = numpy.sum(numpy.stack(
#             [self.noise_generators[i].get_noise_for_signal(time_signal, snr, seed, **kwargs) for i in noise_idx],
#             axis=0), axis=0) / len(noise_idx)
#
#         set_snr(time_signal, noise, snr)
#
#         return noise


if __name__ == "__main__":
    import doctest

    doctest.testmod()
