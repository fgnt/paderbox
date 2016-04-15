# print('Hey there. A documentation is available here:\n'
#       'http://ntjenkins.upb.de/view/PythonToolbox/job/python_toolbox/Documentation/')
from contextlib import suppress

# if mkl is available, set maximum number of threads
with suppress(ImportError):
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())
