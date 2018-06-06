"""
Contains utilities which are mainly useful in a REPL environment.
"""
import itertools
import os
import re
from collections import defaultdict
from pathlib import Path
from pprint import pprint as original_pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from IPython.display import display
from tqdm import tqdm

from nt.io import load_json
from nt.io.play import play
from nt.transform import fbank
from nt.transform import istft
from nt.transform import stft
from nt.visualization import context_manager
from nt.visualization import facet_grid
from nt.visualization import plot

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
    "play",
    "context_manager",
    "facet_grid",
    "plot",
    "fbank",
    "istft",
    "stft",
    "tqdm",
]


def pprint(obj):
    np.set_string_function(
        lambda a: f"array(shape={a.shape}, dtype={a.dtype})"
    )
    original_pprint(obj)
    np.set_string_function(None)
