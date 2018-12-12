"""
Contains utilities which are mainly useful in a REPL environment.
"""
import itertools
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pprint as original_pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML
from IPython.display import display
from nt.database.chime import Chime3, Chime4
from nt.database.reverb import Reverb
from nt.io import dump_json
from nt.io import load_json
from nt.io.play import play
from nt.transform import fbank
from nt.transform import istft
from nt.transform import stft
from nt.utils.numpy_utils import morph
from nt.utils.pandas_helper import py_query
from nt.visualization import context_manager
from nt.visualization import facet_grid
from nt.visualization import plot
from tqdm import tqdm
from nt.database.iterator import AudioReader, LimitAudioLength, AlignmentReader

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
    "context_manager",
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
    "Chime3",
    "Chime4",
    "Reverb",
    "AudioReader",
    "LimitAudioLength",
    "AlignmentReader"
]


def pprint(obj):
    """Shortens the np.ndarray representer."""
    np.set_string_function(
        lambda a: f"array(shape={a.shape}, dtype={a.dtype})"
    )
    original_pprint(obj)
    np.set_string_function(None)
