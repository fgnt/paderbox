
# Does not work in ipynb
# from libcpp.vector cimport vector
# from libcpp.pair cimport pair

# ToDO: better place for testcode
#       http://ntsvr1:1619/notebooks/chime5/2018_05_17_tf_blstm.ipynb

def cy_non_intersection(interval: tuple, intervals: tuple) -> tuple:
    """
    "Removes" one `interval` from `intervals` by removing or shortening any
    intervals in `intervals` that overlap with `interval`.

    Similar to `cy_intersection(inverted_interval, intervals)`.
    """
    cdef:
        long long start
        long long end
        long long i_start
        long long i_end
        list new_interval
    start, end = interval
    new_interval = []

    for i_start, i_end in intervals:

        if start < i_start < end:
            i_start = end
        elif start < i_end < end:
            i_end = start
        elif i_start < start and end < i_end:
            new_interval.append((i_start, start))
            i_start = end

        if i_start < i_end:
            new_interval.append((i_start, i_end))

    return tuple(new_interval)


def cy_intersection(interval, intervals):
    """
    "Cuts" out intervals from `intervals` that lie within `interval`.
    """
    cdef:
        long long start
        long long end
        long long i_start
        long long i_end
        list new_interval

    start, end = interval
    new_interval = []

    for i_start, i_end in intervals:
        i_start = max(start, i_start)
        i_end = min(end, i_end)
        if i_start < i_end:
            new_interval.append((i_start, i_end))

    return tuple(new_interval)

def cy_intersection_double(interval, intervals):
    cdef:
        double start
        double end
        double i_start
        double i_end
        list new_interval

    start, end = interval
    new_interval = []

    for i_start, i_end in intervals:
        i_start = max(start, i_start)
        i_end = min(end, i_end)
        if i_start < i_end:
            new_interval.append((i_start, i_end))

    return tuple(new_interval)

def cy_parse_item(item, shape):

    cdef:
        long long start
        long long end
        long long v
        long long size

    if shape is not None:
        size = shape[-1]
    else:
        size = -1  # dummy assignment for c code

    if not isinstance(item, (slice)):
        raise ValueError(
            f'Expect item ({item}) to have the type slice and not {type(item)}.'
        )
    if item.step is not None:
        raise ValueError(f'Step is not supported {item}')

    # Handle start value
    if item.start is None:
        start = 0
    else:
        start = item.start

    if start < 0:
        if shape is None:
            raise ValueError('Shape has to be given if a negative index is used')
        start = start + size

    if start < 0:
        # Clip the value to 0 to match numpy slice indexing. Numpy also clips
        # the start value at 0 if start < -size. Then, the indexes size doesn't
        # match the size of the slice.
        start = 0

    # Handle stop value
    if item.stop is None:
        if shape is None:
            raise RuntimeError(
                'You tried to slice an ArrayInterval with unknown shape '
                'without a stop value.\n'
                'This is not supported, either the shape has to be known\n'
                'or you have to specify a stop value for the slice '
                '(i.e. array_interval[:stop])\n'
                'You called the array interval with:\n'
                f'    array_interval[{item}]'
            )
        stop = size
    else:
        stop = item.stop

    if stop < 0:
        if shape is None:
            raise ValueError('Shape has to be given if a negative index is used')
        stop = stop + size

    if size > -1:
        # Clip the value at size to match numpy slice indexing. Numpy also clips
        # the stop value at size if stop > size. Then, the indexes size doesn't
        # match the size of the slice.
        if stop > size:
            stop = size

    return start, stop


def cy_str_to_intervals(string):

    cdef:
        str intervals_string
        str interval_string
        str start_str
        str end_str
        long long start
        long long end
        list intervals

    intervals_string = string

#     start, end = interval
    intervals = []

#     for i_start, i_end in intervals:

    for interval_string in intervals_string.replace(' ', '').strip(',').split(','):

        try:
            start_str, end_str = interval_string.split(':')
        except Exception as e:
            print('interval_string in cy_str_to_intervals', repr(interval_string))
            raise Exception(interval_string) from e
        start = int(start_str)
        end = int(end_str)

        intervals.append((start, end))

    return tuple(intervals)


def cy_invert_intervals(normalized_intervals, size):
    """
    Inverts intervals.

    Assumes that the intervals are normalized! This means that:
        - No overlapping intervals in `normalized_intervals`
        - Intervals are sorted by their start times
    """
    cdef:
        list inverted_intervals
        long long edge
        long long i_start
        long long i_end

    if len(normalized_intervals) == 0:
        # Shortcut for emtpy intervals
        return (0, size),

    edge = -1
    inverted_intervals = []
    for i_start, i_end in normalized_intervals:
        if edge == -1:
            if i_start != 0:
                inverted_intervals.append((0, i_start))
        else:
            inverted_intervals.append((edge, i_start))
        edge = i_end

    if edge != size:
        inverted_intervals.append((edge, size))

    return tuple(inverted_intervals)
