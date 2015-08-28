

from pymatbridge import Matlab
from os import environ
from cached_property import cached_property
import unittest

class Mlab():
    @cached_property
    def process(self):
        mlab_process = Matlab('nice -n 3 /net/ssd/software/MATLAB/R2015a/bin/matlab -nodisplay -nosplash')
        mlab_process.start()
        _ = mlab_process.run_code('run /net/home/ldrude/Projects/2015_python_matlab/matlab/startup.m')
        return mlab_process

# define decorator to skip matlab_tests
matlab_test = unittest.skipUnless(environ.get('TEST_MATLAB'),'matlab-test')
