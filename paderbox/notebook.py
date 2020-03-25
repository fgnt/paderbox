"""
Contains utilities which are mainly useful in a REPL environment.
"""
import itertools
import collections
import functools
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint as original_pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import IPython.lib.pretty
from IPython.display import HTML
from IPython.display import display
# from paderbox.database.chime import Chime3, Chime4
# from paderbox.database.reverb import Reverb
from paderbox.io import dump_json
from paderbox.io import load_json
from paderbox.io.play import play
from paderbox.transform import fbank
from paderbox.transform import istft
from paderbox.transform import stft
from paderbox.array import morph
from paderbox.utils.pretty import pprint
from paderbox.utils.pandas_utils import py_query
from paderbox.visualization import figure_context
from paderbox.visualization import axes_context
from paderbox.visualization import facet_grid
from paderbox.visualization import plot
from tqdm import tqdm
# from paderbox.database.iterator import AudioReader
# from paderbox.database.iterator import LimitAudioLength
# from paderbox.database.iterator import AlignmentReader

__all__ = [
    "pprint",
    "itertools",
    "os",
    "re",
    "defaultdict",
    "Path",
    "pd",
    "sns",
    "HTML",
    "display",
    "plt",
    "load_json",
    "dump_json",
    "play",
    "axes_context",
    "figure_context",
    "facet_grid",
    "plot",
    "fbank",
    "istft",
    "stft",
    "tqdm",
    "morph",
    "np",
    "datetime",
    "py_query",
]
