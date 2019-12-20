import os.path
import shutil
import socket
import warnings

from cached_property import cached_property

from paderbox.io.data_dir import matlab_toolbox, matlab_r2015a, matlab_license


class Mlab:
    def __init__(self, matlab_startup_path=None):

        home_startup = os.path.expanduser('~/startup.m')

        if matlab_startup_path is None and os.path.isfile(home_startup):
            self.matlab_startup_path = home_startup
            # Example ~/startup.m:
            #   cd '~/path/to/repo'
            #   startup
        elif matlab_startup_path is None:
            self.matlab_startup_path = \
                str(matlab_toolbox / 'startup.m')
        else:
            self.matlab_startup_path = matlab_startup_path

    @cached_property
    def process(self):
        from pymatbridge import Matlab
        hostname = socket.gethostname()
        if 'nt' in hostname:
            matlab_executable = str(matlab_r2015a / 'bin' / 'matlab')
            mlab_process = Matlab(
                'nice -n 3 ' +
                matlab_executable +
                ' -c {}'.format(matlab_license) +
                ' -nodisplay -nosplash'
            )
        else:
            matlab_executable = 'matlab'
            mlab_process = Matlab(
                f'nice -n 3 {matlab_executable} -nodisplay -nosplash')
        if shutil.which(matlab_executable) is None:
            # pymatbridge.Matlab has a long timeout and will fail with
            # "ValueError: MATLAB failed to start" when matlab does not exists.
            raise EnvironmentError(
                'Could not find matlab.'
            )
        mlab_process.start()
        ret = mlab_process.run_code('run ' + self.matlab_startup_path)
        if not ret['success']:
            print(ret)
            raise NameError('Matlab is not working!')
        return mlab_process

    def run_code(self, code, check_success=True):
        ret = self.process.run_code(code)

        assert code[-1] == ';', \
            'Every single line of Matlab code must end with a semicolon.'

        if check_success and not ret['success']:
            print(ret)
            print(code)
            warnings.warn(str('Matlab code not executeable! \nCode:\t ')
                          + str(code) +
                          str('\nMatlab return:\t ') +
                          str(ret))
            raise NameError('Matlab code not executeable! See above warning.')
        return ret

    def run_code_print(self, code, check_success=True):
        ret = self.run_code(code)
        print('Matlab content: ', ret['content'])

    def get_variable(self, code):
        return self.process.get_variable(code)

    def set_variable(self, code, var):
        return self.process.set_variable(code, var)
