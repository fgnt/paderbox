import numpy as np
# https://github.com/numpy/numpy/issues/11456
# https://github.com/ContinuumIO/anaconda-issues/issues/9697
# https://github.com/IntelPython/mkl_fft/issues/11
np.fft.restore_all()
