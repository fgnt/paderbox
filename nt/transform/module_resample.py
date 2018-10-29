"""
This module contains resampling methods.
"""
import subprocess
from pathlib import Path
import numpy as np


def resample_sox(signal: np.ndarray, *, in_rate, out_rate):
    """Resample using the Swiss Army knife of sound processing programs (SOX).

    This function exists to mimic as closely as possible how resampling would
    be realized in a `wav.scp` file in Kaldi.

    >>> signal = np.array([1, -1, 1, -1], dtype=np.float32)
    >>> resample_sox(signal, in_rate=2, out_rate=1)
    array([ 0.28615326, -0.13513082], dtype=float32)

    Args:
        signal: Signal as one-dimensional np.ndarray: Shape (T,)
        in_rate: Probably as an integer
        out_rate: Probably as an integer

    Returns: Resampled version with same dtype as input.

    """
    assert signal.dtype == np.float32, \
        "The call to SOX just implements float32."

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

    command = [
        'sox',
        '-N',
        '-V1',
        '-t', 'f32',
        '-r', f'{in_rate}',
        '-c', '1',
        '-',
        '-t', 'f32',
        '-r', f'{out_rate}',
        '-c', '1',
        '-'
    ]
    process = subprocess.run(
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=signal.tobytes(order="f")
    )
    signal_resampled = np.fromstring(process.stdout, dtype=signal.dtype)
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
    return signal_resampled
