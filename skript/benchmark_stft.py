"""
ntpc9
OMP_NUM_THREADS 1
MKL_NUM_THREADS 1

nt
3.2839200710877776
[3.289062741678208, 3.2875119652599096, 3.277139882091433]

librosa
1.6728829271160066
[1.6732992688193917, 1.663632761221379, 1.6640413324348629]

scipy
3.2334349588491023
[3.255621672142297, 3.2345176991075277, 3.2378146476112306]

python_speech_features
3.7981791491620243
[3.7941275471821427, 3.8010904281400144, 3.7978039011359215]

"""
import numpy as np
import timeit
import paderbox as pb
import librosa
import scipy.signal
import os
import python_speech_features
from functools import partial
import socket


B = 8
T = 16000 * 5
X = np.random.normal(size=(B, T))
SIZE = 1024
SHIFT = 256


def setup_nt():
    fn = partial(pb.transform.stft, size=SIZE, shift=SHIFT, fading=False, pad=False)

    return X, fn


def setup_librosa():
    # Librosa cache is off by default
    # https://librosa.github.io/librosa/cache.html#enabling-the-cache

    def fn(x_):
        # Does not support independent axis
        return [librosa.stft(x__, n_fft=SIZE, hop_length=SHIFT, center=False) for x__ in x_]

    return X, fn


def setup_scipy():
    fn = partial(scipy.signal.stft, nperseg=SIZE, noverlap=SIZE - SHIFT)
    return X, fn


def setup_python_speech_features():
    def fn1(x_):

        frames = python_speech_features.sigproc.framesig(
            x_, frame_len=SIZE, frame_step=SHIFT
        )
        return np.fft.rfft(frames, SIZE)

    def fn2(x_):
        return [fn1(x__) for x__ in x_]

    return X, fn2


if __name__ == '__main__':
    print(socket.gethostname())
    print('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS'))
    print('MKL_NUM_THREADS', os.environ.get('MKL_NUM_THREADS'))
    print()
    repeats = 100

    for library in 'nt librosa scipy python_speech_features'.split():
        print(library)
        t = timeit.Timer(
            'fn(x)',
            setup=(
                f'from __main__ import setup_{library}; '
                f'x, fn = setup_{library}()'
            )
        )
        print(t.timeit(number=repeats))
        print(t.repeat(number=repeats))
        print()
