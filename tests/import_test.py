"""Test whether all files from paderbox/nt are importable"""
import os
from pathlib import Path
import subprocess
import importlib

import pytest

import paderbox


def get_module_name_from_file(file):
    """
    >> import paderbox as pb
    >> file = pb.transform.module_stft.__file__
    >> file  # doctest: +ELLIPSIS
    '.../paderbox/transform/module_stft.py'
    >> get_module_name_from_file(file)
    'paderbox.transform.module_stft'
    >> file = pb.transform.__file__
    >> file  # doctest: +ELLIPSIS
    '.../paderbox/transform/__init__.py'
    >> get_module_name_from_file(pb.transform.__file__)
    'paderbox.transform'
    """

    # coppied from inspect.getabsfile
    file = os.path.normcase(os.path.abspath(file))
    if os.path.basename(file) == '__init__.py':
        file, _ = os.path.split(file)

    file, module_path = os.path.split(file)
    module_path = os.path.splitext(module_path)[0]
    while file:
        # See setuptools.PackageFinder._looks_like_package
        if not os.path.isfile(os.path.join(file, '__init__.py')):
            break
        file, part = os.path.split(file)
        module_path = part + '.' + module_path
    if '.' in module_path:
        return module_path
    else:
        return '__main__'


class TestImport:
    NT_PATH = Path(paderbox.__file__).parent

    python_files = NT_PATH.glob('**/*.py')

    @pytest.mark.parametrize('py_file', [
            pytest.param(
                py_file,
                id=get_module_name_from_file(py_file))
            for py_file in python_files
    ])
    def test_import(self, py_file, with_importlib=True):
        """
        Import `py_file` into the system
        :param py_file: Python file to import
        :param with_importlib: If True, use `importlib` for importing. Else,
            use `subprocess.run`: It is considerably slower but may have better
             readable test output
        :raise: `AssertionError` if file cannot be imported
        """
        import_name = get_module_name_from_file(py_file)
        suffix = py_file.suffix
        try:
            if with_importlib:
                _ = importlib.import_module(import_name)
            else:
                _ = subprocess.run(['python', '-c',
                                    f'import {import_name}'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=True)
        except (ImportError, ModuleNotFoundError,
                subprocess.CalledProcessError) as e:
            try:
                err = e.stderr.decode('utf-8')
            except AttributeError:
                err = 'See Traceback above'
            assert False, f'Cannot import file "{import_name}{suffix}" \n\n' \
                          f'stderr: {err}'
