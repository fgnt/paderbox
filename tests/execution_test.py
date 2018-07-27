"""Test whether all files from toolbox/nt are importable and executable"""

from pathlib import Path
import os
import inspect
import subprocess
import importlib
import tempfile
import shutil

from nose import SkipTest
from nose.tools import istest
from parameterized import parameterized, param


def _custom_name_func(testcase_func, param_num, param):
    import_name, suffix = _get_import_name(param.args[0], return_suffix=True)
    return f"%s: %s%s" %(
        testcase_func.__name__,
        import_name, suffix
    )


def _get_import_name(py_file, return_suffix=False):
    import_name = '.'.join(py_file.parts[py_file.parts.index('nt'):-1]
                           + (py_file.stem,)
                           )  # convert path to Python's import notation
    if not return_suffix:
        return import_name
    return import_name, py_file.suffix


class TestImportAndExecution:
    
    TOOLBOX_PATH = Path(
        os.path.dirname(  # <path_to_toolbox>/toolbox
            os.path.dirname(  # <path_to_toolbox>/toolbox/tests
                os.path.abspath(inspect.getfile(inspect.currentframe()))
                # <path_to_toolbox>/toolbox/tests/execution_test.py
            )
        )
    )
    NT_PATH = TOOLBOX_PATH / 'nt'

    python_files = NT_PATH.glob('**/*.py')
    python_files = [py_file for py_file in python_files
                    if 'TODO' not in str(py_file)
                    ]
    python_files = list(map(lambda p: (p.parents[0] if p.stem == '__init__'
                                       else p), python_files)
                        )  # replace __init__.py files with their package path
    
    expected_failures = [
        "nt/database/wsj/create_wsj_with_corrected_paths.py",  # no permission
        "nt/database/chime5/get_speaker_activity.py",  # no input arguments
        "nt/database/wsj/write_wav.py",  # no input arguments
        "nt/database/audio_set/clean.py",  # no file to clean
        "nt/database/merl_mixtures_mc/create_files.py"  # no input arguments
    ]

    tmp_dir = tempfile.mkdtemp(dir='.')  # write possible outputs of
    # files into `tmp_dir`

    import_input = list(
        map(lambda p: param(p, with_importlib=True), python_files)
    )

    @parameterized.expand(import_input, doc_func=(lambda func, num, p: None),
                          testcase_func_name=_custom_name_func)
    def test_a_import(self, py_file, with_importlib=True):
        """
        Import `py_file` into the system
        :param py_file: Python file to import
        :param with_importlib: If True, use `importlib` for importing. Else,
            use `subprocess.run`: It is considerably slower but has cleaner
            test output
        :raise: `AssertionError` if file cannot be imported
        """
        import_name = _get_import_name(py_file)
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
            assert False, f'Cannot import file "{import_name}.py" \n\n' \
                          f'stderr: {err}'
  
    execution_input = list(map(lambda p: param(p, timeout=1), python_files))

    @parameterized.expand(execution_input,
                          doc_func=(lambda func, num, p: None),
                          testcase_func_name=_custom_name_func)
    def test_b_execution(self, file, timeout=5):
        """
        Execute `file` with `subprocess.run`
        :param file: File to execute
        :param timeout: Stop execution after `timeout` seconds to have a
            feasible test time. After the timeout, the file is considered to be
            executable. Since `subprocess.run` is slow, `timeout` cannot be
            chosen too low in order to get a reliable test feedback
        :raise: `AssertionError` if process call returned with an error
        """
        cmd = ['python', str(file)]
        if not file.suffix:  # `file` is a package
            raise SkipTest('Do not execute packages')
        try:
            _ = subprocess.run(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=self.tmp_dir, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if any(file.match(expected) for expected in self.expected_failures):
                assert False, 'Expected a failure instead of timeout ' \
                              '(maybe timeout too low?)'
        except subprocess.CalledProcessError as e:
            if not any(file.match(expected) for expected in
                       self.expected_failures):
                assert False, f'Execution of "{str(file)}" failed \n\n' \
                              f'stderr: {e.stderr.decode("utf-8")}'

    @istest
    def z_tear_down(self):
        """
        Remove generated output and tmp_dir
        Test case avoids using teardown mechanism.
        """
        assert os.path.exists(self.tmp_dir), 'Tmp dir was not created!'
        shutil.rmtree(self.tmp_dir)
        assert not os.path.exists(self.tmp_dir), 'Tmp dir was not removed!'
