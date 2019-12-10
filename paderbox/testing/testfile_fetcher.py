import urllib.request as url
import os
from pathlib import Path


def fetch_file_from_url(urlpath, fname):
    """
    Checks if local cache directory possesses an example named <fname>.
    If not found, loads data from urlpath and stores it under <fname>

    Args:
        urlpath: url to the example
        fname: name of the testfile

    Returns: Path to fname

    """
    dirname = os.path.dirname(__file__)
    if not os.path.exists(dirname + "/cache/"):
        os.mkdir(dirname + "/cache/")

    if not os.path.isfile(dirname + "/cache/"+fname):
        path = url.urlopen(urlpath)
        data = path.read()

        with open(dirname + "/cache/" + fname, "wb") as f:
            f.write(data)
    return Path(dirname + "/cache/"+fname)
