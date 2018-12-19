"""Sacred helper when used for local folders with JsonObserver.

Support for MongoDB has been dropped. If you still need it, let us know.
"""
from pathlib import Path
import datetime
from typing import List, Tuple
import warnings
import getpass
from sacred.commands import print_config
from sacred import Experiment, Ingredient


__all__ = [
    'get_dir',
    'print_config',
    'get_path_by_id'
]


def get_dir(_run) -> Path:
    """
    Gets the current directory from this run, e.g. `pth_models/pit/3/`.
    Args:
        _run: From Sacred internal

    Returns:

    """
    assert len(_run.observers) == 1, len(_run.observers)
    _dir = Path(_run.observers[0].basedir) / str(_run._id)
    return _dir


def get_path_by_id(
    data_root: Path,
    experiment_name: str,
    _id: str
):
    """Helps to get full path, if you just know the previous ID.

    Args:
        data_root: Path object, i.e. Path('/net/vol/project/data')
        experiment_name: String, i.e. 'enhancement'
        _id: Desired ID as string, i.e. '58e23bbf6753904febf824eb'
    """
    experiment_path = data_root / experiment_name
    path = next(experiment_path.glob('*{}*'.format(_id)))
    assert path.is_dir(), 'Folder {} for ID {} not found.'.format(path, _id)
    return path
