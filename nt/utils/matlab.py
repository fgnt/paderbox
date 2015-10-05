

from pymatbridge import Matlab
from os import environ
from cached_property import cached_property
import unittest
import socket

class Mlab():
    @cached_property
    def process(self):
        hostname = socket.gethostname()
        if 'nt' in hostname:
            mlab_process = Matlab('nice -n 3 /net/ssd/software/MATLAB/R2015a/bin/matlab -nodisplay -nosplash')
        else:
            mlab_process = Matlab('nice -n 3 matlab -nodisplay -nosplash')
        mlab_process.start()
        ret = mlab_process.run_code('run /net/ssd/software/matlab_toolbox/startup.m')
        if not ret['success']:
            print(ret)
            raise NameError('Matlab is not working!')
        return mlab_process

# define decorator to skip matlab_tests
matlab_test = unittest.skipUnless(environ.get('TEST_MATLAB'),'matlab-test')
