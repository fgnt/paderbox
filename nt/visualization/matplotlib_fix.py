import os
import matplotlib

# Fix matplotlib when no display is available
# https://github.com/ipython/ipython/pull/10974
if os.environ.get('DISPLAY', '') == '':
    if matplotlib.get_backend() in [
        'module://ipykernel.pylab.backend_inline',  # %matplotlib inline
        'NbAgg',  # %matplotlib notebook
    ]:
        # Don't change if server has no DISPLAY but is connected to notebook
        pass
    else:
        matplotlib.use('Agg')
