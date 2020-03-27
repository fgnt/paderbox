"""

Description for rttm files copied from kaldi chime6 receipt
   `steps/segmentation/convert_utt2spk_and_segments_to_rttm.py`:

Each line in an rttm file contains the following values:

    <type> <file-id> <channel-id> <begin-time> \
            <duration> <ortho> <stype> <name> <conf>
    <type> = SPEAKER for each segment.
    <file-id> - the File-ID of the recording
    <channel-id> - the Channel-ID, usually 1
    <begin-time> - start time of segment
    <duration> - duration of segment
    <ortho> - <NA> (this is ignored)
    <stype> - <NA> (this is ignored)
    <name> - speaker name or id
    <conf> - <NA> (this is ignored)

"""

from pathlib import Path
import collections
import decimal

import numpy as np

from paderbox.array.intervall.core import ArrayIntervall, zeros


def _merge_dicts(*dicts):
    if len(dicts) == 1 and isinstance(dicts[0], (tuple, list)):
        dicts = dicts[0]
    keys = [k for d in dicts for k in d.keys()]
    # Duplicate test
    assert len(keys) == len(set(keys)), (keys, dicts)

    return {
        k: v
        for d in dicts
        for k, v in d.items()
    }


def from_rttm(rttm_file, shape=None, sample_rate=16000):
    """

    Args:
        rttm_file:
        shape:
        sample_rate:

    Returns:
        Nested dictionary. The keys of the outer dict will the the file-ids.
        The inner dict has as keys the name (i.e. speaker name or id).
        The values of the inner dict will be an array intervall.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir) / 'dummy.rttm'
    ...     file.write_text("SPEAKER S02 1 0 1 <NA> <NA> 1 <NA>\\nSPEAKER S02 1 2 1 <NA> <NA> 1 <NA>\\nSPEAKER S02 1 0 2 <NA> <NA> 2 <NA>")
    ...     print(file.read_text())
    ...     print(from_rttm(file))
    104
    SPEAKER S02 1 0 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 2 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 0 2 <NA> <NA> 2 <NA>
    {'S02': {'1': ArrayIntervall("0:16000, 32000:48000", shape=None), '2': ArrayIntervall("0:32000", shape=None)}}
    """

    kwargs = dict(shape=shape, sample_rate=sample_rate)

    if isinstance(rttm_file, (tuple, list)):
        return _merge_dicts(*[
            from_rttm(f, **kwargs)
            for f in rttm_file
        ])
    
    rttm_file = Path(rttm_file)
    return from_rttm_str(
        rttm_file.read_text(), **kwargs)


def from_rttm_str(rttm_str, shape=None, sample_rate=16000):

    from paderbox.utils.nested import deflatten
    import decimal

    lines = rttm_str.splitlines()

    # SPEAKER S02_U06.ENH 1   40.60    3.22 <NA> <NA> P05 <NA>

    data = collections.defaultdict(lambda: zeros(shape))

    for line in lines:
        parts = line.split()
        assert parts[0] == 'SPEAKER'
        file_id = parts[1]
        channel_id = parts[2]
        begin_time = decimal.Decimal(parts[3])
        duration_time = decimal.Decimal(parts[4])
        name = parts[7]

        end_time = (begin_time + duration_time) * sample_rate
        begin_time = begin_time * sample_rate

        assert begin_time == int(begin_time), begin_time
        assert end_time == int(end_time), end_time

        data[(file_id, name)][int(begin_time):int(end_time)] = 1

    return deflatten(data, sep=None)


def to_rttm_str(data, sample_rate=16000):
    """

    data: nested dictionary:
     - `data[file_id][speaker_id] = activity`
     - `activity` is an boolian array that indicate activity or not on sample
       resolution (e.g. ArrayIntervall or np.array)


    >>> ar1 = zeros(None)
    >>> ar1[0:16000] = 1
    >>> ar1[32000:48000] = 1
    >>> ar2 = np.zeros(shape=50000, dtype=np.bool)
    >>> ar2[0:32000] = 1
    >>> data = {'S02': {'1': ar1, '2': ar2}}
    >>> data
    {'S02': {'1': ArrayIntervall("0:16000, 32000:48000", shape=None), '2': array([ True,  True,  True, ..., False, False, False])}}
    >>> print(to_rttm_str(data))
    SPEAKER S02 1 0 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 2 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 0 2 <NA> <NA> 2 <NA>
    >>> print(to_rttm_str(data, sample_rate=320_000))
    SPEAKER S02 1 0 0.05 <NA> <NA> 1 <NA>
    SPEAKER S02 1 0.1 0.05 <NA> <NA> 1 <NA>
    SPEAKER S02 1 0 0.1 <NA> <NA> 2 <NA>
    >>> print(to_rttm_str({'S02': [ar1, ar2]}))
    SPEAKER S02 1 0 1 <NA> <NA> 0 <NA>
    SPEAKER S02 1 2 1 <NA> <NA> 0 <NA>
    SPEAKER S02 1 0 2 <NA> <NA> 1 <NA>

    """
    lines = []
    for file_id in data.keys():
        if isinstance(data[file_id], dict):
            keys = data[file_id].keys()
        else:
            keys = range(len(data[file_id]))

        for name in keys:
            content = data[file_id][name]
            if isinstance(content, np.ndarray):
                content = ArrayIntervall(content)
            for begin, end in content.intervals:
                duration = decimal.Decimal(int(end - begin)) / sample_rate
                begin = decimal.Decimal(int(begin)) / sample_rate
                lines.append(
                    f'SPEAKER {file_id} 1 {begin} {duration} <NA> <NA> {name} <NA>'
                )

    return '\n'.join(lines)


def to_rttm(data, rttm_file, sample_rate=16000):
    Path(rttm_file).write_text(to_rttm_str(data, sample_rate=sample_rate))
