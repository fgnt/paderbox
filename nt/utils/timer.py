import time
import datetime
from chainer import cuda

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


def timeStamped(fname, fmt='{fname}_%Y-%m-%d-%H-%M-%S'):

    """ Timestamps a string according to ``fmt``
    :param fname: String to timestamp
    :param fmt: Format of the timestamp where ``{fname}`` is the placeholder for the string
    :return: timestamped string
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
