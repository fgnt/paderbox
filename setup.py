"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
from codecs import open
# Always prefer setuptools over distutils
from distutils.core import setup
from os import path

import numpy
from setuptools import find_packages
from Cython.Build import cythonize

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
try:
    with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description=''

setup(
    name='paderbox',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.0',

    description='Collection of utilities in the nt department',
    long_description=long_description,

    # The project's main homepage.
    url='http://nt.upb.de/',

    # Author details
    author='Department of Communications Engineering',
    author_email='sek@nt.upb.de',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='sample setuptools development',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'tabulate',
        'scipy',
        'seaborn',
        'tqdm',
        'dill',
        "dataclasses ; python_version<'3.7'",  # dataclasses is in py37 buildin
        'pathos',  # Multiprocessing alternative
        'pip',
        'IPython',
        'ipywidgets',
        'scikit-learn',
        'pyzmq',
        # 'pymatbridge',  # need pyzmq to be installed manually
        'h5py',
        'line_profiler',
        'memory_profiler',
        'cached_property',
        'editdistance',
        # 'Pyro4',
        # 'psutil',
        # 'plumbum',
        'click',
        'fire',
        'typecheck-decorator',
        'natsort',
        # 'pymongo',  # Used to generate and check Sacred IDs
        'coverage',  # for nosetests --with-coverage
        'beautifulsoup4',
        'pysoundfile',  # for German speech database
        'wavefile',  # for reading .flac audio
        'nose',
        'parameterized',
        'sh',
        'pycallgraph',  # Used in profiling module
        # 'pafy',  # Download Youtube files for some databases
        'lazy_dataset',  # used for iteration over database examples
        'librosa',  # Used for FBanks
    ],

    # Installation problems in a clean, new environment:
    # 1. `cython` and `scipy` must be installed manually before using
    # `pip install`
    # 2. `pyzmq` has to be installed manually, otherwise `pymatbridge` will
    # complain

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    ext_modules=cythonize([
    ],
        annotate=True,
    ),
    include_dirs=[numpy.get_include()],
    entry_points={
        "console_scripts": [
            'paderbox.strip_solution = paderbox.utils.strip_solution:entry_point',
        ]
    },
)
