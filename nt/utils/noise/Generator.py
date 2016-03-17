import numpy
from abc import ABCMeta, abstractmethod

class NoiseGeneratorTemplate():
    __metaclass__ = ABCMeta
    name = 'Unknown'

    @abstractmethod
    def get_noise_for_signal(self, time_signal, snr, seed = None, **kwargs):
        pass

    def _normalize_noise(self, time_signal, snr, noise_signal):
        signal_power = numpy.var(time_signal)
        noise_power = numpy.var(noise_signal)
        snr_lin = numpy.power(10,snr/10)
        noise_std = numpy.sqrt(signal_power/(snr_lin*noise_power))
        noise =  noise_std * noise_signal
        return noise

class NoiseGeneratorWhite(NoiseGeneratorTemplate):
    name = 'whiteGaussian'

    def __init__(self, snr = None):
        self.snr = snr

    def get_noise_for_signal(self, time_signal, snr, seed = None, **kwargs):
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
        >>> print('%.1f' % SNR)
        20.0

        """
        shape = time_signal.shape
        numpy.random.seed(seed=seed)
        noise_signal = numpy.random.randn(*shape)
        return self._normalize_noise(time_signal,snr,noise_signal)
'''
def _pink_noise_generator(N,D):
    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    A = [1 -2.494956002, 2.017265875, -0.522189400]
    nT60 = round(numpy.log(1000)/(1-max(abs(numpy.roots(A))))) # T60 est.
    v = numpy.random.randn(N+nT60, D) # Gaussian white noise: N(0,1)
    x = filter(B, A, v)    # Apply 1/F roll-off to PSD
    x = x[nT60+1:, :] # Skip transient response
    return(x)
'''


if __name__ == "__main__":
    import doctest
    doctest.testmod()
























