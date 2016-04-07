from IPython.display import display
from IPython.display import Audio
from scipy import signal
from nt.transform import istft
import numpy as np


def play(data, channel=0, rate=16000,
         size=1024, shift=256, window=signal.blackman):
    """ Tries to guess, what the input data is. Plays time series and stft.

    Provides an easy to use interface to play back sound in an IPython Notebook.

    :param data: Time series with shape (frames,)
        or stft with shape (frames, channels, bins) or (frames, bins)
    :param channel: Channel, if you have a multichannel stft signal.
    :param rate: Sampling rate in Hz.
    :param size: STFT window size
    :param shift: STFT shift
    :param window: STFT analysis window
    :return:
    """
    if np.issubdtype(data.dtype, np.complex):
        assert data.shape[-1] == size//2 + 1, 'Wrong number of frequency bins.'

        if len(data.shape) == 3:
            data = data[:, channel, :]

        data = istft(data, size=size, shift=shift, window=window)

    assert np.issubdtype(data.dtype, np.float)
    assert len(data.shape) == 1

    display(Audio(data, rate=rate))
