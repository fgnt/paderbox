from pathlib import Path
import urllib.request as url


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


def url_to_local_path(fpath, file=None):
    """
    Checks if local cache directory possesses an example named <file>.
    If not found, loads data from urlpath and stores it under <fpath>

    Args:
        fpath: url to the example repository
        file: name of the testfile

    Returns: Path to file

    """
    path = get_cache_dir()

    if file is None:
        # remove difficult letters
        file = fpath.replace(':', '_').replace('/', '_')

    if not (path / file).exists():
        datapath = url.urlopen(fpath)
        data = datapath.read()

        with open(path / file, "wb") as f:
            f.write(data)
    return path / file
