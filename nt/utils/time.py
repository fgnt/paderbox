import time
import datetime

class Timer(object):
    """ Time code execution.

    Example usage:
        ```python
        with Timer as t:
            sleep(10)
        print(t.secs)
        ```

    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Timestamps a string according to ``fmt``
    :param fname: String to timestamp
    :param fmt: Format of the timestamp where ``{fname}`` is the placeholder for the string
    :return: timestamped string
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)