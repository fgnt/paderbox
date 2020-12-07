from .interval import *
from .interval import ArrayInterval as ArrayIntervall

import warnings
warnings.warn(
    'Using ArrayIntervall (with double l) from paderbox.array.intervall (with '
    'double l) is deprecated. Use ArrayInterval from paderbox.array.interval '
    '(with a single l) instead.'
)
