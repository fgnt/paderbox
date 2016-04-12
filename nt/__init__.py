# print('Hey there. A documentation is available here:\n'
#       'http://ntjenkins.upb.de/view/PythonToolbox/job/python_toolbox/Documentation/')

# if mkl is available, set maximum number of threads
try:
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())
except ImportError:
    pass

