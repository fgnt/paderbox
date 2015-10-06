

from pymatbridge import Matlab
from os import environ
from cached_property import cached_property
import unittest
import socket
import warnings
import os.path

class Mlab():
    def __init__(self, matlab_startup_path=None):

        homeStartup = os.path.expanduser('~/startup.m')

        if matlab_startup_path == None and os.path.isfile(homeStartup):
            self.matlab_startup_path = homeStartup
            # Example ~/startup.m:
            #   cd '~/path/to/repo'
            #   startup
        elif matlab_startup_path == None:
            self.matlab_startup_path = '/net/ssd/software/matlab_toolbox/startup.m'
        else:
            self.matlab_startup_path = matlab_startup_path



    @cached_property
    def process(self):
        hostname = socket.gethostname()
        if 'nt' in hostname:
            mlab_process = Matlab('nice -n 3 /net/ssd/software/MATLAB/R2015a/bin/matlab -nodisplay -nosplash')
        else:
            mlab_process = Matlab('nice -n 3 matlab -nodisplay -nosplash')
        mlab_process.start()
        ret = mlab_process.run_code('run ' + self.matlab_startup_path)
        if not ret['success']:
            print(ret)
            raise NameError('Matlab is not working!')
        return mlab_process

    def run_code(self, code, check_success=True):
        ret = self.process.run_code(code)
        if check_success and not ret['success']:
            print(ret)
            print(code)
            warnings.warn(str('Matlab code not executeable! \nCode:\t ')
                          +str(code)+
                          str('\nMatlab return:\t ')+
                          str(ret))
            raise NameError('Matlab code not executeable! See above warning.')
        return ret

    def get_variable(self, code):
        return self.process.get_variable(code)


# define decorator to skip matlab_tests
matlab_test = unittest.skipUnless(environ.get('TEST_MATLAB'),'matlab-test')
