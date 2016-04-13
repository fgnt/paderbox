from abc import ABCMeta, abstractmethod

import numpy
from nt.speech_enhancement.noise.utils import set_snr
from scipy.signal import lfilter
from nt.speech_enhancement.noise.spherical_habets import _sinf_3D_py
import nt.testing as tc
from functools import wraps
import nt.io.audioread as ar
from nt.database.noisex92 import helper


class NoiseGeneratorTemplate:
    __metaclass__ = ABCMeta
    name = 'Unknown'

    @abstractmethod
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        pass


def _decorator_noise_generator_set_snr(f):
    """ This decorator sets the seed and fix the snr.

    :param f: Function to be wrapped
    :return: noise_signal
    """
    @wraps(f)
    def wrapper(self, time_signal, snr, seed=None, **kwargs):
        numpy.random.seed(seed=seed)
        noise_signal = f(self, time_signal, snr, seed, **kwargs)
        set_snr(time_signal, noise_signal, snr)
        return noise_signal

    return wrapper


class NoiseGeneratorWhite(NoiseGeneratorTemplate):

    @_decorator_noise_generator_set_snr
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        """
        Example:

        >>> import nt.evaluation.sxr as sxr
        >>> time_signal = numpy.random.randn(1000)
        >>> n_gen = NoiseGeneratorWhite()
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> SNR
        20.0

        >>> import nt.evaluation.sxr as sxr
        >>> time_signal = numpy.random.randn(1000, 2)
        >>> n_gen = NoiseGeneratorWhite()
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n.shape
        (1000, 2)
        >>> time_signal.shape
        (1000, 2)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, numpy.newaxis], n)
        >>> SNR
        20.0

        """
        shape = time_signal.shape
        noise_signal = numpy.random.randn(*shape)
        return noise_signal


class NoiseGeneratorPink(NoiseGeneratorTemplate):

    def __init__(self, sample_axis=-2, channel_axis=-1):
        self.sample_dim = sample_axis
        self.channel_dim = channel_axis

    def _pink_noise_generator(self, n, d):
        """Generates pink noise. You still need to rescale it to your needs.

            This code was taken from the book
            Spectral Audio Signal Processing, by Julius O. Smith III,
            W3K Publishing, 2011, ISBN 978-0-9745607-3-1:
            `Spectral Audio Signal Processing <https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html>`

            Alternative implementation can be found in
            `dsp.stackexchange.com <http://dsp.stackexchange.com/a/376/8515>`.

        :param n: Number of samples
        :param d: Number of channels
        :return: Pink noise of dimension number of samples times one
        """

        B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        A = [1, - 2.494956002, 2.017265875, -0.522189400]
        nT60 = 1430 #  T60 est.- Original Matlab Code: nT60 = round(log(1000)/(1-max(abs(roots(A)))));
        v = numpy.random.randn(n + nT60, d)  # Gaussian white noise: N(0,1)
        x = lfilter(B, A, v, axis = 0)  # Apply 1/F roll-off to PSD
        x = x[nT60:, :]  # Skip transient response
        return x

    @_decorator_noise_generator_set_snr
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        """
        Example:


        >>> import nt.evaluation.sxr as sxr
        >>> time_signal = numpy.random.randn(1000)
        >>> n_gen = NoiseGeneratorPink()
        >>> pinknoise = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n.shape
        (1000,)
        >>> time_signal.shape
        (1000,)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None])
        >>> SNR
        20

        >>> import nt.evaluation.sxr as sxr
        >>> time_signal = numpy.random.randn(1000, 5)
        >>> n_gen = NoiseGeneratorPink()
        >>> pinknoise = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n.shape
        (1000, 5)
        >>> time_signal.shape
        (1000, 5)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n)
        >>> SNR
        20.0

         """
        if len(time_signal.shape) > 1:
            n = time_signal.shape[self.sample_dim]
            d = time_signal.shape[self.channel_dim]
            noise_signal = self._pink_noise_generator(n, d)
        else:
            n = time_signal.shape[0]
            d = 1
            noise_signal = self._pink_noise_generator(n, d)[:, 0]

        return noise_signal


class NoiseGeneratorNoisex92(NoiseGeneratorTemplate):
    # last_label = None

    def __init__(self, label=None, sample_rate=16000):

        self.labels = helper.get_labels()
        if label is not None:
            if label not in self.labels:
                raise KeyError('The label {label} does not exist. '
                               'Please choose a valid label from following list: '
                               '{l}'.format(label=label, l=', '.join(self.labels)))
            self.labels = [label]
        self.audio_datas = list()
        for l in self.labels:
                path = helper.get_path_for_label(l, sample_rate)
                self.audio_datas += [ar.audioread(path, sample_rate=sample_rate)]
        self.sample_rate = sample_rate

    @_decorator_noise_generator_set_snr
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        """

        Example:

        >>> import nt.evaluation.sxr as sxr
        >>> time_signal = numpy.random.randn(1000)
        >>> seed = 1
        >>> label = 'destroyerengine'
        >>> label = 'destroyerengin'
        >>> n_gen = NoiseGeneratorNoisex92(sample_rate = 16000)
        >>> n = n_gen.get_noise_for_signal(time_signal, 20, seed=1)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> SNR
        20
        >>> n = n_gen.get_noise_for_signal(time_signal, 20, seed=2)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
        >>> SNR
        20
        """
        idx = numpy.random.choice(len(self.audio_datas))
        audio_data = self.audio_datas[idx]
        # self.last_label = self.labels[idx]
        print(self.labels[idx])
        if time_signal.shape[0] <= audio_data.shape[0]:
            seq = numpy.random.randint(0, audio_data.shape[0] - time_signal.shape[0])
        else:
            raise ValueError('')
        noise_signal = audio_data[seq: seq + time_signal.shape[0]]
        return noise_signal


class NoiseGeneratorSpherical(NoiseGeneratorTemplate):

    def __init__(self, sensor_positions, *, sample_axis=-2, channel_axis=-1, sample_rate=16000, c=340,
                 number_of_cylindrical_angels=256):

        assert sensor_positions.shape[0] == 3

        self.sample_axis = sample_axis
        self.channel_axis = channel_axis

        self.sensor_positions = sensor_positions
        _, self.number_of_channels = sensor_positions.shape

        self.sample_rate = sample_rate
        self.sound_velocity = c
        self.number_of_cylindrical_angels = number_of_cylindrical_angels

    @_decorator_noise_generator_set_snr
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        """
        Example:

        >>> import nt.evaluation.sxr as sxr
        >>> from nt.utils.math_ops import sph2cart
        >>> time_signal = numpy.random.randn(1000, 3)
        >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
        >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
        >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
        >>> n_gen = NoiseGeneratorSpherical(P)
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, None], n[:, :, None])
        >>> SNR
        20.0

        >>> import nt.evaluation.sxr as sxr
        >>> from nt.utils.math_ops import sph2cart
        >>> time_signal = numpy.random.randn(1000, 3)
        >>> x1,y1,z1 = sph2cart(0,0,0.1)    # Sensor position 1
        >>> x2,y2,z2 = sph2cart(0,0,0.2)    # Sensor position 2
        >>> P = numpy.array([[0, x1, x2], [0, y1, y2], [0, z1, z2]]) # Construct position matrix
        >>> n_gen = NoiseGeneratorSpherical(P)
        >>> n = n_gen.get_noise_for_signal(time_signal, 20)
        >>> n.shape
        (1000, 3)
        >>> time_signal.shape
        (1000, 3)
        >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, :, numpy.newaxis], n)
        >>> SNR
        20.0

        """
        tc.assert_equal(time_signal.shape[self.channel_axis], self.number_of_channels)

        # shape = time_signal.shape
        noise_signal = _sinf_3D_py(self.sensor_positions, time_signal.shape[self.sample_axis])
        return noise_signal
        # return self._normalize_noise(time_signal,snr,noise_signal)


class NoiseGeneratorMix:
    """

    Example:

    >>> import nt.evaluation.sxr as sxr
    >>> time_signal = numpy.random.randn(1000)
    >>> n_gens = [NoiseGeneratorWhite(), NoiseGeneratorPink()]
    >>> n_gen = NoiseGeneratorMix(n_gens, max_different_types=2)
    >>> n = n_gen.get_noise_for_signal(time_signal, 20)
    >>> SDR, SIR, SNR = sxr.input_sxr(time_signal[:, None, None], n[:, None, None])
    >>> SNR
    20.0


    """

    def __init__(self, noise_generators, max_different_types=1, probabilities=None):
        self.noise_generators = noise_generators
        for n in noise_generators:
            assert isinstance(n, NoiseGeneratorTemplate)
        self.max_different_types = max_different_types
        if probabilities:
            self.probabilities = probabilities / sum(probabilities)
        else:
            self.probabilities = numpy.ones(len(noise_generators)) * 1 / len(noise_generators)

    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):

        numpy.random.seed(seed=seed)
        # replace = True ; Ziehen mit zurücklegen
        # replace = False ; Ziehen ohne zurücklegen
        noise_idx = numpy.random.choice(len(self.noise_generators), self.max_different_types, replace=True,
                                        p=self.probabilities)

        noise = numpy.sum(numpy.stack(
            [self.noise_generators[i].get_noise_for_signal(time_signal, snr, seed, **kwargs) for i in noise_idx],
            axis=0), axis=0) / len(noise_idx)

        set_snr(time_signal, noise, snr)

        return noise


if __name__ == "__main__":
    import doctest

    doctest.testmod()
