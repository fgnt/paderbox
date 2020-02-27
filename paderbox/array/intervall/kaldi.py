from pathlib import Path
import collections
import decimal

import numpy as np

from paderbox.array.intervall.core import zeros
from paderbox.array.intervall.rttm import _merge_dicts


def from_kaldi_segments(segments_file, shape=None, sample_rate=16000, round_fn=None):
    kwargs = dict(shape=shape, sample_rate=sample_rate, round_fn=round_fn)
    if isinstance(segments_file, (tuple, list)):
        return _merge_dicts(*[
            from_kaldi_segments(f, **kwargs)
            for f in segments_file
        ])

    rttm_file = Path(segments_file)
    return from_kaldi_segments_str(
        rttm_file.read_text(), **kwargs)


def from_kaldi_segments_str(segments_str, shape=None, sample_rate=16000, round_fn=None):
    """

    >>> s = 'S02_U06.ENH-0004121-0004187 S02_U06.ENH 41.21 41.87\\n'
    >>> s += 'S02_U06.ENH-0010122-0010337 S02_U06.ENH 101.23 103.37\\n'
    >>> from_kaldi_segments_str(s, sample_rate=1000)
    {'S02_U06.ENH': ArrayIntervall("41210:41870, 101230:103370", shape=None)}

    >>> s = 'S02_U06.ENH-0004121-0004187 S02_U06.ENH 41.21 41.87\\n'
    >>> s += 'S02_U06.ENH-0010122-0010337 S02_U06.ENH BUG 101.23 103.37\\n'
    >>> from_kaldi_segments_str(s, sample_rate=1000)
    Traceback (most recent call last):
    ...
    ValueError: Expect "<uttID> <fileID> <start> <end>".
    Got ['S02_U06.ENH-0010122-0010337', 'S02_U06.ENH', 'BUG', '101.23', '103.37']

    """
    from paderbox.utils.nested import deflatten

    lines = segments_str.splitlines()

    # Example:
    # Utterance-ID File-ID Start End
    # S02_U06.ENH-0004121-0004187 S02_U06.ENH 41.21 41.87

    data = collections.defaultdict(lambda: zeros(shape))

    for line in lines:
        parts = line.split()
        #
        try:
            _, file_id, begin_time, end_time = parts
        except ValueError:
            raise ValueError(
                'Expect "<uttID> <fileID> <start> <end>".\n'
                f'Got {parts}') from None
        begin_time = decimal.Decimal(begin_time)
        end_time = decimal.Decimal(end_time)
        if round_fn:
            begin_time = round_fn(begin_time)
            end_time = round_fn(end_time)

        end_time = end_time * sample_rate
        begin_time = begin_time * sample_rate

        assert begin_time == int(begin_time), begin_time
        assert end_time == int(end_time), end_time

        data[file_id][int(begin_time):int(end_time)] = 1

    return dict(data)
