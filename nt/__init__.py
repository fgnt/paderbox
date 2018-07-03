from numpy.fft import restore_all as _restore_all
# https://github.com/numpy/numpy/issues/11456
# https://github.com/ContinuumIO/anaconda-issues/issues/9697
# https://github.com/IntelPython/mkl_fft/issues/11
_restore_all()
