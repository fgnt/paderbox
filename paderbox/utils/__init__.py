from paderbox.utils.timer import Timer, timeStamped

from .deprecated import deprecated

__all__ = [
    'deprecated',
    'dtw',
    'lazy_parallel_map',
    'mapping',
    'matlab',
    'misc',
    'mpi',
    'numpy_utils',
    'options',
    'pandas_utils',
    'pc2',
    'process_caller',
    'random_utils',
    'sacred_utils',
    'strip_solution',
    'timer',
    'transcription_handling',
]

# Lazy import all subpackages
# Note: define all subpackages in __all__
import sys
import pkgutil
import operator
import importlib.util

_available_submodules = list(map(
    operator.itemgetter(1),
    pkgutil.iter_modules(__path__)
))


class _LazySubModule(sys.modules[__name__].__class__):
    # In py37 is the class is not nessesary and __dir__ and __getattr__ are enough
    # https://snarky.ca/lazy-importing-in-python-3-7/

    def __dir__(self):
        ret = super().__dir__()
        return [*ret, *_available_submodules]

    def __getattr__(self, item):
        if item in _available_submodules:
            import importlib
            return importlib.import_module(f'{__package__}.{item}')
        else:
            return super().__getattr__(item)


sys.modules[__name__].__class__ = _LazySubModule

del sys, pkgutil, operator, importlib, _LazySubModule
