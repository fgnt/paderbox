"""Test whether all files from toolbox/nt are importable and executable"""

from pathlib import Path
import os
import inspect
import subprocess
import importlib
import tempfile
import shutil

from nose import SkipTest


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
    failed_on_import = []
    
    expected_failures = [
        "nt/database/wsj/create_wsj_with_corrected_paths.py",  # no permission
        "nt/database/chime5/get_speaker_activity.py",  # no input arguments
        "nt/database/wsj/write_wav.py",  # no input arguments
        "nt/database/audio_set/clean.py",  # no file to clean
        "nt/database/merl_mixtures_mc/create_files.py"  # no input arguments
    ]
    
    # This test needs to run first to remove files that cannot be imported from
    # execution test. `nose` runs the test in alphabetical order so 'test_a_...'
    # will run before 'test_b_...'
    def test_a_import(self):
        for py_file in self.python_files:
            # This generates sub-tests with nosetests
            yield self.check_import, py_file, True

    def check_import(self, py_file, with_importlib=True):
        """
        Import `py_file` into the system
        :param py_file: Python file to import
        :param with_importlib: If True, use `importlib` for importing. Else,
            use `subprocess.run`: It is considerably slower but has cleaner
            test output
        :raise: `AssertionError` if file cannot be imported
        """
        import_name = '.'.join(py_file.parts[py_file.parts.index('nt'):-1]
                               + (py_file.stem,)
                               )  # convert path to Python's import notation
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
            self.failed_on_import.append(py_file)
            try:
                err = e.stderr.decode('utf-8')
            except AttributeError:
                err = 'See Traceback above'
            assert False, f'Cannot import file "{import_name}.py" \n\n' \
                          f'stderr: {err}'

    def test_b_execution(self):
        tmp_dir = tempfile.mkdtemp(dir='.')  # write possible outputs of files
        # into `tmp_dir`
        to_execute = set(self.python_files) - set(self.failed_on_import)
        for file in to_execute:
            yield self.check_execution, file, tmp_dir, 10
        shutil.rmtree(tmp_dir)  # remove generated output and `tmp_dir`

    def check_execution(self, file, cwd, timeout=5):
        """
        Execute `file` with `subprocess.run`
        :param file: File to execute
        :param cwd: Path to directory where file will be executed. Some scripts
            may create new files. Executing in `cwd` will simplify clean-up
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
                               cwd=cwd, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if any(file.match(expected) for expected in self.expected_failures):
                assert False, 'Expected a failure instead of timeout ' \
                              '(maybe timeout too low?)'
        except subprocess.CalledProcessError as e:
            if not any(file.match(expected) for expected in
                       self.expected_failures):
                assert False, f'Execution of "{str(file)}" failed \n\n' \
                              f'stderr: {e.stderr.decode("utf-8")}'
