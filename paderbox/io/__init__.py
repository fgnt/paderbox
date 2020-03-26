"""
This module deals with all sorts of input and output.

There is special focus on audio, but there are also some convenience imports
i.e. for load_json() and similar functions.

The file path is called `path` just as it has been done in ``audioread``.
The path should either be a ``pathlib.Path`` object or a string.
"""
import pickle
from pathlib import Path

from paderbox.io.path_utils import normalize_path
from paderbox.io.atomic import (
    open_atomic,
    write_text_atomic,
    write_bytes_atomic,
)
from paderbox.io import audioread
from paderbox.io import hdf5
from paderbox.io import play
from paderbox.io.json_module import (
    load_json,
    loads_json,
    dump_json,
    dumps_json,
)
from paderbox.io.json_module import SummaryEncoder
from paderbox.io.yaml_module import (
    load_yaml,
    loads_yaml,
    load_yaml_unsafe,
    loads_yaml_unsafe,
    dump_yaml,
    dumps_yaml,
    dump_yaml_unsafe,
    dumps_yaml_unsafe,
)
from paderbox.io.audioread import load_audio, recursive_load_audio
from paderbox.io.audiowrite import dump_audio, dumps_audio
from paderbox.io.file_handling import (
    mkdir_p,
    symlink,
)
from paderbox.io import data_dir
from .wrapper_load import load
from .wrapper_dump import dump

__all__ = [
    "load_audio",
    "recursive_load_audio",
    "dump_audio",
    "dumps_audio",
    "load_json",
    "loads_json",
    "dump_json",
    "dumps_json",
    "load_yaml",
    "loads_yaml",
    "load_yaml_unsafe",
    "loads_yaml_unsafe",
    "dump_yaml",
    "dumps_yaml",
    "dump_yaml_unsafe",
    "dumps_yaml_unsafe",
    "load_hdf5",
    "dump_hdf5",
    "update_hdf5",
    "load_pickle",
    "dump_pickle",
    "mkdir_p",
    "symlink",
    "SummaryEncoder",
    "data_dir",
]


def load_hdf5(path, internal_path="/"):
    # ToDo: drop this wrapper
    path = normalize_path(path, as_str=True, allow_fd=False)
    return hdf5.load_hdf5(path, str(internal_path))


def dump_hdf5(data, path):
    # ToDo: drop this wrapper
    path = normalize_path(path, as_str=True, allow_fd=True)
    return hdf5.dump_hdf5(data, path)


def update_hdf5(data, path, prefix="/"):
    # ToDo: drop this wrapper
    assert isinstance(path, (str, Path, hdf5.h5py.File))
    if isinstance(path, hdf5.h5py.File):
        hdf5.update_hdf5(data, path, path=prefix)
    else:
        path = normalize_path(normalize_path, as_str=True, allow_fd=False)
        hdf5.update_hdf5(data, path, path=prefix)


def load_pickle(path):
    path = normalize_path(path, allow_fd=False)
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(data, path):
    path = normalize_path(path, allow_fd=False)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
