
import os
from pathlib import Path

from paderbox.kaldi.helper import get_kaldi_env
import pytest

@pytest.mark.parametrize("path",
                         list(set(get_kaldi_env()['PATH'].split(':')) -
                              set(os.environ['PATH'].split(':')))
)
def test_get_kaldi_env_path(path):
    assert Path(path).exists(), path