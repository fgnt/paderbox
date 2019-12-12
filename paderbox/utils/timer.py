import time
import datetime

from collections import defaultdict


class Timer(object):
    """ Time code execution.

    Example usage::

        with Timer() as t:
            sleep(10)
        print(t)

    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.secs = None
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        if self.verbose:
            print(self)

    def __repr__(self):
        if self.secs is None:
            return f'{self.__class__}: running...'
        else:
            return f'{self.__class__}: {self.secs:.3g} s'


class TimerDictEntry:

    def __init__(self, style):
        self.time = 0.0
        self._style = style
        self.timestamp = time.perf_counter  # time.process_time
        self._start = []

    def __enter__(self):
        self._start.append(self.timestamp())
        return self

    def __exit__(self, *args):
        end = self.timestamp()
        self.time += end - self._start.pop(-1)

    def __call__(self, iterable):
        it = iter(iterable)
        try:
            while True:
                with self:
                    x = next(it)
                yield x
        except StopIteration:
            pass

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

    # Wrap iterator
    >>> def slow_range(N):
    ...     for i in range(N):
    ...         time.sleep(0.1)
    ...         yield i
    >>> for i in t['test_3'](slow_range(3)):
    ...    pass

    # Nesting
    >>> with t['test_4']:
    ...     time.sleep(1)
    ...     with t['test_4']:
    ...         pass

    # Show the timings
    >>> times = t.as_dict
    >>> sorted(times.keys())
    ['test', 'test_2', 'test_3', 'test_4']
    >>> print('\\n'.join([f'{k} ' + str(times[k])[:9]
    ...                   for k in sorted(times.keys())]))
    test 0:00:02.0
    test_2 0:00:01.0
    test_3 0:00:00.3
    test_4 0:00:01.0
    """
    _display_data = None
    _display_vbox = None

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

    def print_as_yaml(self, file=None):
        if file is None:
            print('Times are in seconds')
        print(self.as_yaml, file=None)

    def __repr__(self):
        return 'TimerDict: ' + self.as_dict.__repr__()

    def __str__(self):
        return self.as_dict.__str__()

    @property
    def widget(self):
        import ipywidgets

        def gen():
            l = [
                ipywidgets.widgets.FloatProgress(
                    min=0,
                    max=1,
                    step=0.1,
                    bar_style='info',
                    orientation='horizontal'
                ),
                ipywidgets.Label('')
            ]
            return ipywidgets.HBox(l)

        self._display_data = defaultdict(gen)
        self._display_vbox = ipywidgets.VBox()

        self.widget_update()
        return self._display_vbox

    def widget_update(self):
        assert self._display_data is not None, 'Did you forget "display(timer_dict.widget)" ?'

        d = {k: v.total_seconds() for k, v in self.as_dict.items()}
        total = sum([v for v in d.values()])

        for k, v in d.items():
            float_progress, label = self._display_data[k].children
            float_progress.value = v
            float_progress.description = str(k)
            label.value = str(v)
            float_progress.max = total

        if len(self._display_vbox.children) != len(self._display_data):
            self._display_vbox.children = [
                v for k, v in self._display_data.items()
            ]


def timeStamped(fname, fmt='{fname}_%Y-%m-%d-%H-%M-%S'):
    """ Timestamps a string according to ``fmt``
    :param fname: String to timestamp
    :param fmt: Format of the timestamp where ``{fname}`` is the placeholder for the string
    :return: timestamped string
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)
