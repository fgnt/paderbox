from pathlib import Path
import collections

from paderbox.array.intervall.core import zeros


def ArrayIntervalls_from_rttm(rttm_file, shape=None, sample_rate=16000):
    """
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     file = Path(tmpdir) / 'dummy.rttm'
    ...     file.write_text("SPEAKER S02 1 0 1 <NA> <NA> 1 <NA>\\nSPEAKER S02 1 2 1 <NA> <NA> 1 <NA>\\nSPEAKER S02 1 0 2 <NA> <NA> 2 <NA>")
    ...     print(file.read_text())
    ...     print(ArrayIntervalls_from_rttm(file))
    104
    SPEAKER S02 1 0 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 2 1 <NA> <NA> 1 <NA>
    SPEAKER S02 1 0 2 <NA> <NA> 2 <NA>
    {'S02': {'1': ArrayIntervall("0:16000, 32000:48000", shape=None), '2': ArrayIntervall("0:32000", shape=None)}}
    """

    # Description for rttm files copied from kaldi chime6 receipt
    #    `steps/segmentation/convert_utt2spk_and_segments_to_rttm.py`:
    # <type> <file-id> <channel-id> <begin-time> \
    #         <duration> <ortho> <stype> <name> <conf>
    # <type> = SPEAKER for each segment.
    # <file-id> - the File-ID of the recording
    # <channel-id> - the Channel-ID, usually 1
    # <begin-time> - start time of segment
    # <duration> - duration of segment
    # <ortho> - <NA> (this is ignored)
    # <stype> - <NA> (this is ignored)
    # <name> - speaker name or id
    # <conf> - <NA> (this is ignored)
    from paderbox.utils.nested import deflatten
    import decimal

    rttm_file = Path(rttm_file)
    lines = rttm_file.read_text().splitlines()

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

        assert begin_time == int(begin_time)
        assert end_time == int(end_time)

        data[(file_id, name)][int(begin_time):int(end_time)] = 1

    return deflatten(data, sep=None)
