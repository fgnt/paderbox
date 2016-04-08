from abc import ABCMeta, abstractmethod

import numpy

from nt.speech_enhancement.noise.utils import set_snr
from scipy.signal import lfilter


class NoiseGeneratorTemplate:
    __metaclass__ = ABCMeta
    name = 'Unknown'

    @abstractmethod
    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        pass


class NoiseGeneratorWhite(NoiseGeneratorTemplate):
    name = 'whiteGaussian'

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
        numpy.random.seed(seed=seed)
        noise_signal = numpy.random.randn(*shape)
        set_snr(time_signal, noise_signal, snr)
        return noise_signal
        # return self._normalize_noise(time_signal,snr,noise_signal)


class NoiseGeneratorPink(NoiseGeneratorTemplate):
    name = 'pinkNoise'

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
        x = x[nT60 :, :]  # Skip transient response
        return (x)

    def get_noise_for_signal(self, time_signal, snr, seed=None, **kwargs):
        """
        Example:

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
        numpy.random.seed(seed=seed)
        n = time_signal.shape[self.sample_dim]
        if len(time_signal.shape) > 1:
            d = time_signal.shape[self.channel_dim]
        else:
            d = 1
        noise_signal = self._pink_noise_generator(n, d)
        set_snr(time_signal, noise_signal, snr)
        return noise_signal


class NoiseGeneratorMix:
    """

    Example:

    >>> import nt.evaluation.sxr as sxr
    >>> time_signal = numpy.random.randn(1000)
    >>> n_gens = [NoiseGeneratorWhite() for _ in range(2)] # stupid example, makes no sens
    >>> n_gen = NoiseGeneratorMix(n_gens)
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
            [self.noise_generators[noise_idx].get_noise_for_signal(time_signal, snr, seed, **kwargs) for i in noise_idx]
            , axis=0), axis=0) / len(noise_idx)

        return noise


if __name__ == "__main__":
    import doctest

    doctest.testmod()
