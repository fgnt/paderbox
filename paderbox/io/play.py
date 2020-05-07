import sys
import os
from pathlib import Path

from IPython.display import display
from IPython.display import Audio
import numpy as np
from scipy import signal

from paderbox.io.audioread import load_audio
from paderbox.transform import istft


class NamedAudio(Audio):
    name = None

    def _repr_html_(self):
        audio_html = super()._repr_html_()

        assert self.name is not None

        return """
        <table style="width:100%">
            <tr>
                <td style="width:25%">
                    {}
                </td>
                <td style="width:75%">
                    {}
                </td>
            </tr>
        </table>
        """.format(self.name, audio_html)


def play(
        data,
        channel=0,
        sample_rate=16000,
        size=1024,
        shift=256,
        window=signal.blackman,
        window_length: int=None,
        *,
        scale=1,
        name=None,
        stereo=False,
        normalize=True,
):
    """ Tries to guess, what the input data is. Plays time series and stft.

    Provides an easy to use interface to play back sound in an IPython Notebook.

    :param data: Time series with shape (frames,)
        or stft with shape (frames, channels, bins) or (frames, bins)
        or string containing path to audio file.
    :param channel: Channel, if you have a multichannel stft signal or a
        multichannel audio file.
    :param sample_rate: Sampling rate in Hz.
    :param size: STFT window size
    :param shift: STFT shift
    :param window: STFT analysis window
    :param scale: Scale the Volume, currently only amplification with clip
        is supported.
    :param name: If name is set, then in ipynb table with name and audio is
                 displayed
    :param stereo: If set to true, you can listen to channel as defined by
        `channel` parameter and the next channel at the same time.
    :param normalize: It true, normalize the data to have values in the range
        from 0 to 1. Can only be disabled with newer IPython versions.
    :return:
    """
    if isinstance(data, dict):
        assert name is None, name
        for k, v in data.items():
            play(
                data=v,
                name=k,

                channel=channel,
                sample_rate=sample_rate,
                size=size,
                shift=shift,
                window=window,
                window_length=window_length,
                scale=scale,
                stereo=stereo,
            )
        return

    if stereo:
        if isinstance(channel, int):
            channel = (channel, channel+1)
    else:
        assert isinstance(channel, int), (type(channel), channel)

    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, str):
        assert os.path.exists(data), ('File does not exist.', data)
        data = load_audio(data, expected_sample_rate=sample_rate)
        if len(data.shape) == 2:
            data = data[channel, :]
    elif np.iscomplexobj(data):
        assert data.shape[-1] == size // 2 + \
            1, ('Wrong number of frequency bins', data.shape, size)

        if len(data.shape) == 3:
            data = data[:, channel, :]

        data = istft(
            data,
            size=size,
            shift=shift,
            window=window,
            window_length=window_length,
        )
    elif np.isrealobj(data):
        if len(data.shape) == 2:
            data = data[channel, :]

    assert np.isrealobj(data), data.dtype
    assert stereo or len(data.shape) == 1, data.shape

    if scale != 1:
        assert scale > 1 or (not normalize), \
            'Only Amplification with clipping is supported. \n' \
            'Note: IPython.display.Audio scales the input, therefore a ' \
            'np.clip can increase the power, but not decrease it. ' \
            f'scale={scale}'
        max_abs_data = np.max(np.abs(data))
        data = np.clip(data, -max_abs_data/scale, max_abs_data/scale)

    if stereo:
        assert len(data.shape) == 2, data.shape
        assert data.shape[0] == 2, data.shape

    if normalize:
        # ToDo: disable this version specific check
        # ipython 7.3.0 has no `normalize` argument and normalize couldn't
        # be disabled
        kwargs = {}
    else:
        # ipython 7.12.0 `Audio` has a `normalize` argument see
        # https://github.com/ipython/ipython/pull/11650
        kwargs = {'normalize': normalize}

    if name is None:
        display(Audio(data, rate=sample_rate, **kwargs))
    else:
        na = NamedAudio(data, rate=sample_rate, **kwargs)
        na.name = name
        display(na)


# Allows to use `paderbox.io.play` instead of `paderbox.io.play.play`
class MyModule(sys.modules[__name__].__class__):
    __call__ = staticmethod(play)


sys.modules[__name__].__class__ = MyModule
