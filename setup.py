"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# To use a consistent encoding
import sys
from codecs import open
# Always prefer setuptools over distutils
from distutils.core import setup
from os import path

import numpy
from setuptools import find_packages
from Cython.Build import cythonize

here = path.abspath(path.dirname(__file__))

# visualization specific dependencies
visualization = ['seaborn', 'IPython', 'ipywidgets',
                 'beautifulsoup4', 'tabulate']
# dependencies only required during test
test = [
    'pytest',
    'pytest-cov',
    'torch',
    'coverage',
    'h5py',
    'pyyaml>=5.1',  # See https://msg.pyyaml.org/load
    'librosa',  # Used for FBanks
    'wavefile',  # for reading .flac audio
    'dill',
    'pathos',
    # 'pyzmq',
    # 'pymatbridge',  # need pyzmq to be installed manually
    'tqdm',
    'fire',
    'pycallgraph',  # Used in profiling module
    (
        # line_profiler does not work in python 3.7
        # https://github.com/rkern/line_profiler/issues/132
        # 'line_profiler; python_version<"3.7"'
        'line_profiler'
        if sys.version_info < (3, 7) else
        # Install from repo works also in py37:
        # 'line_profiler @ git+https://github.com/rkern/line_profiler; python_version>="3.7"'
        'line_profiler @ git+https://github.com/rkern/line_profiler'
        # `; python_version<"3.7"` does not work with `git+...`
    ),
    'memory_profiler',
]

# Get the long description from the relevant file
try:
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name='paderbox',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.3',

    description='Collection of utilities in the department of communications engineering of the UPB',
    long_description=long_description,

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # The project's main homepage.
    url='https://github.com/fgnt/paderbox/',

    # Author details
    author='Department of Communications Engineering, Paderborn University',
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
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='audio speech',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'numpy',
        'scipy',
        "dataclasses; python_version<'3.7'",  # dataclasses is in py37 buildin
        'soundfile',
        'cached_property',
    ],

    # Installation problems in a clean, new environment:
    # 1. `cython` and `scipy` must be installed manually before using
    # `pip install`

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[all]
    extras_require={
        'visualization': visualization,
        'all': test + visualization,
    },

    ext_modules=cythonize([
        'paderbox/array/intervall/util.pyx',
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
