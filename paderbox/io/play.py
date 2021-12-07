import sys
import os
import functools
from pathlib import Path

import numpy as np

from paderbox.io.audioread import load_audio


def play(
        data,
        channel=0,
        sample_rate=16000,
        size=1024,
        shift=256,
        window='blackman',
        window_length: int=None,
        *,
        scale=1,
        name=None,
        stereo=False,
        normalize=True,
        display=True,
):
    """
    Tries to guess, what the input data is. Plays time series and stft.

    Provides an easy to use interface to play back sound in an IPython Notebook.

    Args:
        data: Time series with shape (frames,)
            or stft with shape (frames, channels, bins) or (frames, bins)
            or string containing path to audio file.
        channel: Channel, if you have a multichannel stft signal or a
            multichannel audio file.
        sample_rate: Sampling rate in Hz.
        size: STFT window size
        shift: STFT shift
        window: STFT analysis window
        window_length: STFT window_length
        scale: Scale the Volume, currently only amplification with clip
            is supported.
        name: If name is set, then in ipynb table with name and audio is
            displayed
        stereo: If set to true, you can listen to channel as defined by
            `channel` parameter and the next channel at the same time.
        normalize: It true, normalize the data to have values in the range
            from 0 to 1. Can only be disabled with newer IPython versions.
        display: When True, display the audio, otherwise return the widget.
    Returns:

    >>> import paderbox as pb
    >>> pb.io.play(np.array([1, 2, 3]), display=False)
    <IPython.lib.display.Audio object>
    >>> pb.io.play({'a': np.array([1, 2, 3]), 'b': np.array([1, 2, 3])}, display=False)
    {'a': <paderbox.io.play.NamedAudio object>, 'b': <paderbox.io.play.NamedAudio object>}

    """
    if isinstance(data, dict):
        assert name is None, name
        kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ['data', 'name']
        }
        audio = {
            k: play(data=v, name='.'.join(k), **kwargs )
            for k, v in data.items()
        }
        if display:
            return
        else:
            return audio

    if stereo:
        if isinstance(channel, int):
            channel = (channel, channel+1)
    else:
        assert isinstance(channel, int), (type(channel), channel)

    if isinstance(data, Path):
        data = str(data)

    if isinstance(data, str):
        assert os.path.exists(data), ('File does not exist.', data)
        data, sample_rate = load_audio(data, return_sample_rate=True)
        if len(data.shape) == 2:
            data = data[channel, :]
    elif np.iscomplexobj(data):
        from paderbox.transform import istft

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

    import IPython.display
    if name is None:
        audio = IPython.display.Audio(data, rate=sample_rate, **kwargs)
    else:
        class NamedAudio(IPython.display.Audio):
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
        audio = NamedAudio(data, rate=sample_rate, **kwargs)
        audio.name = name

    if display:
        IPython.display.display(audio)
    else:
        return audio


class Play:
    def __init__(
            self,
            channel=0,
            sample_rate=16000,
            size=1024,
            shift=256,
            window='blackman',
            window_length: int=None,
            *,
            scale=1,
            name=None,
            stereo=False,
            normalize=True,
            display=True,
    ):
        """
        See `pb.io.play`. This is a wrapper around `pb.io.play`, where you can
        change the defaults.

        Args:
            channel: Channel, if you have a multichannel stft signal or a
                multichannel audio file.
            sample_rate: Sampling rate in Hz.
            size: STFT window size
            shift: STFT shift
            window: STFT analysis window
            window_length: STFT window_length
            scale: Scale the Volume, currently only amplification with clip
                is supported.
            name: If name is set, then in ipynb table with name and audio is
                displayed
            stereo: If set to true, you can listen to channel as defined by
                `channel` parameter and the next channel at the same time.
            normalize: It true, normalize the data to have values in the range
                from 0 to 1. Can only be disabled with newer IPython versions.
            display: When True, display the audio, otherwise return the widget.

        Returns:

        >>> play = Play()
        >>> play
        Play(sample_rate=16000)
        >>> play = Play(sample_rate=8000)
        >>> play
        Play(sample_rate=8000)
        >>> play = Play(sample_rate=8000, display=False)
        >>> play
        Play(sample_rate=8000, display=False)
        """
        self.kwargs = locals()
        del self.kwargs['self']

    def __repr__(self):
        import inspect
        sig = inspect.signature(self.__class__)
        parameters = ', '.join([
            f'{k}={self.kwargs[k]!r}'
            for k, v in sig.parameters.items()
            if self.kwargs[k] != v.default or k == 'sample_rate'
        ])
        return f'{self.__class__.__name__}({parameters})'

    @functools.wraps(play)
    def __call__(self, data, *args, **kwargs):
        """
        Tries to guess, what the input data is. Plays time series and stft.

        Provides an easy to use interface to play back sound in an IPython Notebook.

        Args:
            data: Time series with shape (frames,)
                or stft with shape (frames, channels, bins) or (frames, bins)
                or string containing path to audio file.
            channel: Channel, if you have a multichannel stft signal or a
                multichannel audio file.
            sample_rate: Sampling rate in Hz.
            size: STFT window size
            shift: STFT shift
            window: STFT analysis window
            window_length: STFT window_length
            scale: Scale the Volume, currently only amplification with clip
                is supported.
            name: If name is set, then in ipynb table with name and audio is
                displayed
            stereo: If set to true, you can listen to channel as defined by
                `channel` parameter and the next channel at the same time.
            normalize: It true, normalize the data to have values in the range
                from 0 to 1. Can only be disabled with newer IPython versions.
            display: When True, display the audio, otherwise return the widget.

        Returns:

        """
        return play(data, *args, **{**self.kwargs, **kwargs})


# Allows to use `paderbox.io.play` instead of `paderbox.io.play.play`
class MyModule(sys.modules[__name__].__class__):
    __call__ = staticmethod(play)


sys.modules[__name__].__class__ = MyModule
