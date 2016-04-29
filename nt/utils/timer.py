import time
import datetime
from chainer import cuda
from collections import defaultdict, namedtuple
from contextlib import contextmanager

class Timer(object):
    """ Time code execution.

    Example usage::

        with Timer as t:
            sleep(10)
        print(t.secs)

    """
    def __init__(self, cuda_event=False, verbose=False):
        self.verbose = verbose
        self.secs = 0
        self.msecs = 0
        self.start = 0
        self.end = 0
        self.cuda_event = cuda_event

    def __enter__(self):
        if self.cuda_event:
            self.start = cuda.cupy.cuda.Event()
            self.end = cuda.cupy.cuda.Event()
            self.start.record()
            return self
        else:
            self.start = time.time()
            return self

    def __exit__(self, *args):

        if self.cuda_event:
            self.end.record()
            self.end.synchronize()
            self.msecs = cuda.cupy.cuda.get_elapsed_time(self.start, self.end)
            self.secs = self.msecs / 1000
        else:
            self.end = time.time()
            self.secs = self.end - self.start
            self.msecs = self.secs * 1000  # millisecs
            if self.verbose:
                print('elapsed time: %f ms' % self.msecs)


class TimerAccumulateDict(object):
    """
    >>> t = TimerAccumulateDict()
    >>> with t['test']:
    ...     time.sleep(1)
    >>> with t['test']:
    ...     time.sleep(1)
    >>> with t['test_2']:
    ...     time.sleep(1)
    >>> times = t.as_dict
    >>> sorted(times.keys())
    ['test', 'test_2']
    >>> print('test: {:.3} ms, test_2: {:.3} ms'.format(times['test'], times['test_2']))
    test: 2e+03 ms, test_2: 1e+03 ms
    """

    def __init__(self, cuda_event=False, verbose=False, stat=False):

        self.verbose = verbose
        self.cuda_event = cuda_event
        if not stat:
            self.timings = defaultdict(lambda: 0)
        else:
            self.timings = defaultdict(list)
        self.stat = stat

    @contextmanager
    def __getitem__(self, index):
        t = Timer()

        try:
            t.__enter__()
            yield t
        finally:
            t.__exit__()
            if not self.stat:
                self.timings[index] += t.msecs
            else:
                self.timings[index].sec += [t.msecs]

    @property
    def as_dict(self):
        return dict(self.timings)

    @property
    def as_yaml(self):
        import yaml
        return yaml.dump(dict(self.timings), default_flow_style=False)

    def __str__(self):
        return str(dict(self.timings))


def timeStamped(fname, fmt='{fname}_%Y-%m-%d-%H-%M-%S'):

    """ Timestamps a string according to ``fmt``
    :param fname: String to timestamp
    :param fmt: Format of the timestamp where ``{fname}`` is the placeholder for the string
    :return: timestamped string
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
