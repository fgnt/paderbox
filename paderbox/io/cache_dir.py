from pathlib import Path


def get_cache_dir() -> Path:
    """
    Returns a cache dir that can be used as a local buffer.
    Example use cases are intermediate results and downloaded files.

    At the moment this will be a folder that lies in the root of this package.
    Reason for this location and against `~/.cache/paderbox` are:
     - This repo may lie on an faster filesystem.
     - The user has limited space in the home directory.
     - You may install paderbox multiple times and do not want to get side
       effects.
     - When you delete this repo, also the cache is deleted.

    Returns:
        Path to a cache dir

    >>> get_cache_dir()  # doctest: +SKIP
    PosixPath('.../paderbox/cache')

    """
    dirname = Path(__file__).resolve().absolute().parents[2]
    path = dirname / "cache"
    if not path.exists():
        path.mkdir()

    return path
