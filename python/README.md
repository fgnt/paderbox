Install steps
-------------
1. Run `make` to create the dynamic library
2. Make sure you installed NumPy and Cython (`pip install numpy Cython`)
3. Run `python3 setup.py install` (Python 2 might work as well, but I never tested for it.)
4. Currently the setup.py fails to copy the dynamic library to a place where Python can find it. As a workaround you can add the python folder to your library path. On OS X this is done by appending it to the `DYLD_FALLBACK_LIBRARY_PATH` environment variable.

The module is just a prototype...