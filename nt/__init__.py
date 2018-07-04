from numpy.fft import __file__ as FFT_FILE

# https://github.com/numpy/numpy/issues/11456
# https://github.com/ContinuumIO/anaconda-issues/issues/9697
# https://github.com/IntelPython/mkl_fft/issues/11
with open(FFT_FILE) as f:
    if 'patch_fft = True' in f.read():
        raise Exception(
            'Your Numpy version uses MKL-FFT. That version causes '
            f'segmentation faults. To fix it, open {FFT_FILE} and edit '
            'it such that `patch_fft = True` becomes `patch_fft = False`.'
        )
