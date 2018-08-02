"""Test whether all files from toolbox/nt are importable"""

from pathlib import Path
import inspect
import subprocess
import importlib

from parameterized import parameterized, param


def _custom_name_func(testcase_func, _, param):
    import_name = _get_import_name(param.args[0])
    # TODO: Need save testcase function name for Jenkins.
    sep = '_'  # ': ' works
    return f"%s%s%s" % (
        testcase_func.__name__,
        sep,
        import_name
    )


def _get_import_name(py_file, return_suffix=False):
    """
    Convert path to Python's import notation, i.e. "x.y.z"
    :param py_file: Path object to python file
    :param return_suffix: If True, return additionally `py_file.suffix`. Either
        '.py' or '' (if `py_file` is path to a package, i.e. '__init__.py')
    :return:
    """
    if py_file.stem == '__init__':
        py_file = py_file.parents[0]  # replace __init__.py with package path
    import_name = '.'.join(py_file.parts[py_file.parts.index('nt'):-1] +
                           (py_file.stem,)
                           )
    if not return_suffix:
        return import_name
    return import_name, py_file.suffix


class TestImport:
    TOOLBOX_PATH = Path(
        inspect.getfile(inspect.currentframe())
    ).absolute().parents[1]
    NT_PATH = TOOLBOX_PATH / 'nt'

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
        import_name, suffix = _get_import_name(py_file, return_suffix=True)
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
