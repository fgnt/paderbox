from setuptools import setup
from Cython.Build import cythonize

# python3 setup.py build_ext --inplace

setup(
    name='pyrirgen',

    ext_modules=cythonize(['pyrirgen.pyx'
                           ],
                          annotate=True,
                          )

)
