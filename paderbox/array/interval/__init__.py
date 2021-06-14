"""
This folder contains the class ArrayInterval.

The ArrayInterval is very similar to a boolean 1 dimensional numpy array.
It should work as a replacement where such a numpy array could be used.
The advantage of this class is, that it has a memory efficient storage, when
the 1d array represents intervals. This class only stores the slice boundaries,
instead of the values.

The motivation to write this class was to store the voice/source/speech
activity information (e.g. there is speech from sample 16000 to sample 48000)
of a long audio file (> 2h) in memory.
"""
from .core import zeros, ones
from .core import ArrayInterval
from .core import ArrayInterval_from_str as from_str

from .rttm import from_rttm
from .rttm import from_rttm_str
from .rttm import to_rttm
from .rttm import to_rttm_str
from .kaldi import from_kaldi_segments
from .kaldi import from_kaldi_segments_str
