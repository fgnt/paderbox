from setuptools import setup
from Cython.Build import cythonize

# python3 setup.py build_ext --inplace

setup(
    name='rirgen',

    ext_modules=cythonize(['rirgen/pyrirgen.pyx'
                           ],
                          annotate=True,
                          )

)
