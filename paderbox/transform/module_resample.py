"""
This module contains resampling methods.
"""
import subprocess
import numpy as np


def resample_sox(signal: np.ndarray, *, in_rate, out_rate):
    """Resample using the Swiss Army knife of sound processing programs (SoX).

    This function exists to mimic as closely as possible how resampling would
    be realized in a `wav.scp` file in Kaldi.

    We assume SoX version v14.4.2.

    Note:
        SoX version v14.4.1 produces different values.
        In https://github.com/chimechallenge/chime6-synchronisation/commit/cfc11aa8e8ad48c914b594216f4196fbb73a8998
        is tested, that v14.4.1 produces different results.

    SoX does not apply dithering when you do not change the bits per sample.
    We here fixed input and output bitrate, thus, you are fine.

    SoX automatically does a delay compensation and uses linear filters. This
    is already a sane choice.

    >>> signal = np.array([1, -1, 1, -1], dtype=np.float32)
    >>> resample_sox(signal, in_rate=2, out_rate=1)
    array([ 0.28615332, -0.13513082], dtype=float32)

    >>> signal = np.array([1, -1, 1, -1], dtype=np.float32)
    >>> resample_sox(signal, in_rate=1, out_rate=1)
    array([ 1., -1.,  1., -1.], dtype=float32)

    >>> signal = np.random.normal(size=(2, 30)).astype(np.float32)
    >>> a = resample_sox(signal[0], in_rate=1, out_rate=2)
    >>> b = resample_sox(signal[1], in_rate=1, out_rate=2)
    >>> c = resample_sox(signal, in_rate=1, out_rate=2)
    >>> np.testing.assert_allclose([a, b], c)

    >>> signal = np.random.normal(size=(20, 30)).astype(np.float32)
    >>> c = resample_sox(signal, in_rate=1, out_rate=2)

    Args:
        signal: Signal as one-dimensional np.ndarray: Shape (T,)
        in_rate: Probably as an integer
        out_rate: Probably as an integer

    Returns: Resampled version with same dtype as input.

    """
    assert signal.dtype == np.float32, (
        f"The call to SOX just has float32, but signal.dtype={signal.dtype}."
    )
    # assert signal.ndim == 1, f"signal.ndim={signal.ndim} but only supports 1."

    # sox --help
    # -V[LEVEL]              Increment or set verbosity level (default 2); levels:
    #                          1: failure messages
    #                          2: warnings
    #                          3: details of processing
    #                          4-6: increasing levels of debug messages
    # -t|--type FILETYPE     File type of audio
    # -N|--reverse-nibbles   Encoded nibble-order
    # -c|--channels CHANNELS Number of channels of audio data; e.g. 2 = stereo
    # -r|--rate RATE         Sample rate of audio

    if signal.ndim == 1:
        channels = 1
        has_channel = False
    elif signal.ndim == 2:
        channels = signal.shape[-2]
        assert channels < 30, (
            "More channels than expected:\n"
            f"channels={channels}, signal.shape={signal.shape}"
        )
        has_channel = True
    else:
        raise NotImplementedError(signal.shape)

    assert np.maximum(in_rate / out_rate, out_rate / in_rate) < 10, (
        'Schmalenstroer recommends limited resampling factor. If you really '
        'need this, you need to resample in steps. Up to this point it is '
        'unclear if sox automatically does multi-stage resampling.'
    )

    # This rescaling is necessary since SOX introduces clipping, when the
    # input signal is much too large.
    # We normalize each channel independently to avoid rounding errors leading
    # to the channel doc test above to fail randomly.
    normalizer = 0.95 / np.max(np.abs(signal), keepdims=True, axis=-1)
    signal = normalizer * signal

    # See this page for a parameter explanation:
    # https://explainshell.com/explain?cmd=sox+-N+-V1+--type+f32+--rate+16000+--channels+2+-+--type+f32+--rate+8000+--channels+2+-
    command = [
        'sox',
        '-N',
        '-V1',
        '--type', 'f32',
        '--rate', f'{in_rate}',
        '--channels', str(channels),
        '-',
        '--type', 'f32',
        '--rate', f'{out_rate}',
        '--channels', str(channels),
        '-'
    ]
    process = subprocess.run(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=signal.tobytes(order="f")
    )
    signal_resampled = np.frombuffer(process.stdout, dtype=signal.dtype)
    assert signal_resampled.size > 0, (
        'The command did not yield any output:\n'
        f'signal.shape: {signal.shape}\n'
        f'signal_resampled.shape: {signal_resampled.shape}\n'
        f'command: {command}\n'
        f'stdout: {process.stdout.decode()}\n'
        f'stderr: {process.stderr.decode()}\n'
        'Check that sox is installed.\n'
        'OSX: brew update && brew install sox'
    )

    if has_channel:
        signal_resampled = signal_resampled.reshape(channels, -1, order='F')

    return signal_resampled / normalizer

resample = resample_sox
