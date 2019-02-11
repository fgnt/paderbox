
import os
from pathlib import Path
from parameterized import parameterized

from paderbox.kaldi.helper import get_kaldi_env


@parameterized(
    set(get_kaldi_env()['PATH'].split(':')) - set(os.environ['PATH'].split(':'))
)
def test_get_kaldi_env_path(path):
    assert Path(path).exists(), path
