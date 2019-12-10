"""Test whether all files from paderbox/nt are importable"""
import os
from pathlib import Path
import inspect
import subprocess
import importlib

from parameterized import parameterized, param


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


def _custom_name_func(testcase_func, _, param):
    import_name = get_module_name_from_file(param.args[0])
    import_name = '_'.join(import_name.split('.'))
    return f"%s_%s" % (
        testcase_func.__name__,
        import_name
    )


class TestImport:
    TOOLBOX_PATH = Path(
        inspect.getfile(inspect.currentframe())
    ).absolute().parents[1]
    NT_PATH = TOOLBOX_PATH / 'paderbox'

    python_files = NT_PATH.glob('**/*.py')
    python_files = [py_file for py_file in python_files
                    if 'TODO' not in str(py_file)
                    ]

    test_input = list(
        map(lambda p: param(p, with_importlib=True), python_files)
    )

    @parameterized.expand(test_input, doc_func=(lambda func, num, p: None),
                          testcase_func_name=_custom_name_func)
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
