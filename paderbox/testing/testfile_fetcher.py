import urllib.request as url
from pathlib import Path


def fetch_file_from_url(fpath, file):
    """
    Checks if local cache directory possesses an example named <file>.
    If not found, loads data from urlpath and stores it under <fpath>

    Args:
        fpath: url to the example repository
        file: name of the testfile

    Returns: Path to file

    """

    dirname = Path.cwd()
    path = dirname / "cache"
    if not path.exists():
        path.mkdir()

    if not (path/file).exists():
        datapath = url.urlopen(fpath)
        data = datapath.read()

        with open(path/file, "wb") as f:
            f.write(data)
    return path/file


def get_file_path(file):
    """
    Looks up path to a test audio file and returns to the local file.

    Args:
        file: audio file needed for the test

    Returns: Path to audio test file

    """
    _PESQ = "https://github.com/ludlows/python-pesq/raw/master/audio/"
    _PB_BSS = "https://github.com/fgnt/pb_test_data/raw/master/bss_data/" \
              "low_reverberation/"

    url = {
        'sample.wav': _PB_BSS+"speech_source_0.wav",
        'speech.wav': _PESQ+"speech.wav",
        "speech_bab_0dB.wav": _PESQ+"speech_bab_0dB.wav",
    }[file]

    return fetch_file_from_url(url, file)
