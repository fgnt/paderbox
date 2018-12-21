from numpy.fft import __file__ as FFT_FILE

# https://github.com/numpy/numpy/issues/11456
# https://github.com/ContinuumIO/anaconda-issues/issues/9697
# https://github.com/IntelPython/mkl_fft/issues/11
with open(FFT_FILE) as f:
    if 'patch_fft = True' in f.read():
        raise Exception(
            'Your Numpy version uses MKL-FFT. That version causes '
            f'segmentation faults. To fix it, open {FFT_FILE} and edit '
            'it such that `patch_fft = True` becomes `patch_fft = False`.'
        )

__all__ = [
    'database',
    'evaluation',
    'io',
    'kaldi',
    'label_handling',
    'math',
    'speech_enhancement',
    'speech_recognition',
    'testing',
    'TODO',
    'transform',
    'utils',
    'visualization',
]


def _lazy_import_submodules(__path__, __name__, __package__):
    # Lazy import all subpackages
    # Note: define all subpackages in __all__
    import sys
    import pkgutil
    import operator
    import importlib

    _available_submodules = list(map(
        operator.itemgetter(1),
        pkgutil.iter_modules(__path__)
    ))

    class _LazySubModule(sys.modules[__name__].__class__):
        # In py37 is the class  not nessesary and __dir__ and __getattr__ are
        # enough.
        # See: https://snarky.ca/lazy-importing-in-python-3-7

        def __dir__(self):
            ret = super().__dir__()
            return [*ret, *_available_submodules]

        def __getattr__(self, item):
            if item in _available_submodules:
                return importlib.import_module(f'{__package__}.{item}')
            else:
                return super().__getattr__(item)

    sys.modules[__name__].__class__ = _LazySubModule


_lazy_import_submodules(
    __name__=__name__, __path__=__path__, __package__=__package__
)
