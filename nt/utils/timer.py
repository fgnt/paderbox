import time
import datetime

from collections import defaultdict
from cached_property import cached_property


class Timer(object):
    """ Time code execution.

    Example usage::

        with Timer() as t:
            sleep(10)
        print(t.secs)

    """
    @cached_property
    def cuda(self):
        from chainer import cuda
        return cuda

    def __init__(self, cuda_event=False, verbose=False):
        self.verbose = verbose
        self.secs = 0
        self.msecs = 0
        self.start = 0
        self.end = 0
        self.cuda_event = cuda_event

    def __enter__(self):
        if self.cuda_event:
            self.start = self.cuda.cupy.cuda.Event()
            self.end = self.cuda.cupy.cuda.Event()
            self.start.record()
            return self
        else:
            self.start = time.time()
            return self

    def __exit__(self, *args):

        if self.cuda_event:
            self.end.record()
            self.end.synchronize()
            self.msecs = self.cuda.cupy.cuda.get_elapsed_time(
                self.start, self.end)
            self.secs = self.msecs / 1000
        else:
            self.end = time.time()
            self.secs = self.end - self.start
            self.msecs = self.secs * 1000  # millisecs
            if self.verbose:
                print('elapsed time: %f ms' % self.msecs)


class TimerDictEntry:

    def __init__(self, style):
        self.time = 0.0
        self._style = style

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, *args):
        end = time.perf_counter()
        self.time += end - self._start

    def __call__(self, iterable):
        it = iter(iterable)
        while True:
            with self:
                x = next(it)
            yield x

    @property
    def timedelta(self):
        return datetime.timedelta(seconds=self.time)

    @property
    def value(self):
        if self._style == 'timedelta':
            return self.timedelta
        elif self._style == 'float':
            return self.time
        else:
            raise ValueError(self._style)

    def __repr__(self):
        return str(self.value)


class TimerDict:
    """
    >>> t = TimerDict()
    >>> with t['test']:
    ...     time.sleep(1)
    >>> with t['test']:
    ...     time.sleep(1)
    >>> with t['test_2']:
    ...     time.sleep(1)
    >>> def slow_range(N):
    ...     for i in range(N):
    ...         time.sleep(0.1)
    ...         yield i
    >>> for i in t['test_3'](slow_range(3)):
    ...    pass
    >>> times = t.as_dict
    >>> sorted(times.keys())
    ['test', 'test_2', 'test_3']
    >>> print(str(times['test'])[:9], str(times['test_2'])[:9], str(times['test_3'])[:9])
    0:00:02.0 0:00:01.0 0:00:00.3

    """

    def __init__(self, style: ('timedelta', 'float')='timedelta'):
        """
        :param style: default timedelta, alternative float
        """
        self.timings = defaultdict(lambda: TimerDictEntry(style))

    def __getitem__(self, item):
        assert isinstance(item, str)
        return self.timings[item]

    @property
    def as_dict(self):
        return {k: time.value for k, time in self.timings.items()}

    @property
    def as_yaml(self):
        import yaml
        return yaml.dump({k: str(time) for k, time in self.timings.items()},
                         default_flow_style=False)

    def print_as_yaml(self):
        print('Times are in seconds')
        print(self.as_yaml)

    def __repr__(self):
        return 'TimerDict: ' + self.as_dict.__repr__()

    def __str__(self):
        return self.as_dict.__str__()


def timeStamped(fname, fmt='{fname}_%Y-%m-%d-%H-%M-%S'):

    """ Timestamps a string according to ``fmt``
    :param fname: String to timestamp
    :param fmt: Format of the timestamp where ``{fname}`` is the placeholder for the string
    :return: timestamped string
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
