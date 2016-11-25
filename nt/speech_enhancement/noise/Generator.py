import json
from collections import Iterable
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.signal import lfilter

import nt.testing as tc
import nt.io.audioread as ar
from nt.io.audioread import audioread
from nt.speech_enhancement.noise.utils import set_snr
from nt.speech_enhancement.noise.spherical_habets import _sinf_3D_py
from nt.database.noisex92 import helper


class NoiseGeneratorTemplate:
    """
    Make sure, your implementation of get_noise() uses the provided random
    number generator.
    Setting the random number generator state globally is discouraged.
    """
    __metaclass__ = ABCMeta

    def get_noise_for_signal(
            self,
            time_signal,
            *,
            snr,
            rng_state: np.random.RandomState=np.random,
            **kwargs
    ):
        """
        Args:
            time_signal:
            snr: SNR or single speaker SNR.
            rng_state: A random number generator object or np.random
            **kwargs:
        """
        noise_signal = self.get_noise(
            *time_signal.shape,
            rng_state=rng_state,
            **kwargs
        )
        set_snr(time_signal, noise_signal, snr)
        return noise_signal

    def get_noise(self, *shape, rng_state=np.random, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        return self._get_noise(shape=shape, rng_state=rng_state, **kwargs)

    @abstractmethod
    def _get_noise(self, shape, rng_state=np.random):
        pass


class NoiseGeneratorWhite(NoiseGeneratorTemplate):
    """
    Example:

    >>> from nt.evaluation.sxr import input_sxr
    >>> time_signal = np.random.normal(size=(1000,))
    >>> ng = NoiseGeneratorWhite()
    >>> n = ng.get_noise_for_signal(time_signal, snr=20)
    >>> SDR, SIR, SNR = input_sxr(time_signal[:, None, None], n[:, None, None])
    >>> round(SNR, 1)
    20.0

    >>> from nt.evaluation.sxr import input_sxr
    >>> time_signal = np.random.normal(size=(1000, 2))
    >>> ng = NoiseGeneratorWhite()
    >>> n = ng.get_noise_for_signal(time_signal, snr=20)
    >>> n.shape
    (1000, 2)

    >>> time_signal.shape
    (1000, 2)

    >>> SDR, SIR, SNR = input_sxr(time_signal[:, :, None], n)
    >>> round(SNR, 1)
    20.0
    """
    def _get_noise(self, shape, rng_state=np.random):
        noise_signal = rng_state.normal(size=shape)
        return noise_signal


class NoiseGeneratorChimeBackground(NoiseGeneratorTemplate):
    """ Generate random background noise by sampling from Chime background.

    Shape of your input signal is assumed to be (channels, samples).
    Possible leading dimensions have to be singleton.
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
        """

        Args:
            json_src: Path to your chime background data json.
            flist: Either ``all``, ``train``, or ``cv``.
            sampling_rate:
            max_channels: Chose a number of channels from {1, ..., 6}.
        """
        flist = 'all' if flist is None else flist
        with open(json_src) as f:
            database = json.load(f)

        self.sampling_rate = sampling_rate
        self.max_channels = max_channels
        self.flist = database[flist]
        self.utterance_list = sorted(self.flist['wav'].keys())

    def _get_noise(self, shape, rng_state=np.random):
        D, T = shape[-2:]
        assert np.prod(shape[:-2]) == 1

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

        # Reshape to deal with singleton dimensions
        return np.stack(noise_list).reshape(shape)


class NoiseGeneratorPink(NoiseGeneratorTemplate):
    """
    See example code of ``NoiseGeneratorWhite``.
    """
    def _get_noise(self, shape, rng_state=np.random):
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
        v = rng_state.randn(*size)

        # Apply 1/F roll-off to PSD
        noise = lfilter(b, a, v, axis=-1)

        # Skip transient response
        noise = noise[..., transient_samples:]
        return noise


class NoiseGeneratorNoisex92(NoiseGeneratorTemplate):
    def __init__(self, label=None, sample_rate=16000):

        self.labels = helper.get_labels()
        if label is not None:
            if label not in self.labels:
                raise KeyError('The label "{label}" does not exist. '
                               'Please choose a valid label from following list: '
                               '{l}'.format(label=label, l=', '.join(self.labels)))
            self.labels = [label]
        self.audio_datas = list()
        for l in self.labels:
                path = helper.get_path_for_label(l, sample_rate)
                self.audio_datas += [ar.audioread(path, sample_rate=sample_rate)]

        self.sample_rate = sample_rate

    def _get_noise(self, shape, rng_state=np.random, **kwargs):
        """

        Example:

        >>> import nt.evaluation.sxr as sxr, numpy as np
        >>> time_signal = np.random.randn(16000)
        >>> ng = NoiseGeneratorNoisex92(sample_rate = 16000)
        >>> n = ng.get_noise_for_signal(time_signal, snr=20, rng_state=np.random.RandomState(1))
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> '{:.2f}'.format(SNR)
        '20.00'
        >>> n = ng.get_noise_for_signal(time_signal, snr=20)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> '{:.2f}'.format(SNR)
        '20.00'
        >>> label = 'destroyereng'
        >>> ng = NoiseGeneratorNoisex92(label, sample_rate = 16000)
        Traceback (most recent call last):
        ...
        KeyError: 'The label "destroyereng" does not exist. Please choose a valid label from following list: babble, buccaneer1, buccaneer2, destroyerengine, destroyerops, f16, factory1, factory2, hfchannel, leopard, m109, machinegun, pink, volvo, white'
        >>> label = 'destroyerengine'
        >>> ng = NoiseGeneratorNoisex92(label, sample_rate = 16000)
        >>> n = ng.get_noise_for_signal(time_signal, snr=20)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> '{:.2f}'.format(SNR)
        '20.00'
        """
        idx = rng_state.choice(len(self.audio_datas))
        audio_data = self.audio_datas[idx]
        # self.last_label = self.labels[idx]
        if shape[0] <= audio_data.shape[0]:
            seq = rng_state.randint(0, audio_data.shape[0] - shape[0])
        else:
            raise ValueError('Length of Time Signal is longer then the length of a possible noisex92 Noise Signal!')
        noise_signal = audio_data[seq: seq + shape[0]]
        return noise_signal


class NoiseGeneratorSpherical(NoiseGeneratorTemplate):

    def __init__(self, sensor_positions, *, sample_axis=-1, channel_axis=-2,
                 sample_rate=16000, c=340, number_of_cylindrical_angels=256):

        assert sensor_positions.shape[0] == 3

        self.sample_axis = sample_axis
        self.channel_axis = channel_axis

        self.sensor_positions = sensor_positions
        _, self.number_of_channels = sensor_positions.shape

        self.sample_rate = sample_rate
        self.sound_velocity = c
        self.number_of_cylindrical_angels = number_of_cylindrical_angels

    def _get_noise(self, shape, rng_state=np.random):
        """
        Example:

        >>> import nt.evaluation.sxr as sxr, numpy
        >>> from nt.math.vector import sph2cart
        >>> time_signal = numpy.random.randn(1000, 3)
        >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
        >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
        >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
        >>> n_gen = NoiseGeneratorSpherical(P)
        >>> n = n_gen.get_noise_for_signal(time_signal, snr=20)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n[:, :, None])
        >>> '{:.2f}'.format(SNR)
        '20.00'

        >>> time_signal = numpy.random.randn(1000, 3)
        >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
        >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
        >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
        >>> n_gen = NoiseGeneratorSpherical(P)
        >>> n = n_gen.get_noise_for_signal(time_signal, snr=20)
        >>> n.shape
        (1000, 3)
        >>> time_signal.shape
        (1000, 3)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, numpy.newaxis], n)
        >>> '{:.2f}'.format(SNR)
        '20.00'


        """
        tc.assert_equal(shape[self.channel_axis], self.number_of_channels)

        noise_signal = _sinf_3D_py(self.sensor_positions, shape[self.sample_axis])
        return noise_signal.T


if __name__ == "__main__":
    import doctest

    doctest.testmod()
