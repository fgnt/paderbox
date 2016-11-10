import unittest
import numpy as np
import nt.testing as tc
import tempfile
import nose_parameterized
import os
import warnings

from nt.utils.hdf5_utils import hdf5_dump, hdf5_load, hdf5_update, Hdf5DumpWarning


class TestHdf5:

    dir = None

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.__enter__()
        warnings.filterwarnings('error')

    def tearDown(self):
        self.dir = self._dir.__exit__(None, None, None)

    @nose_parameterized.parameterized([
        ('int', {'key': 1}, np.int64(1)),
        ('float', {'key': 1.1}, np.float64(1.1)),
        ('complex', {'key': 1.1j}, np.complex128(1.1j)),
        ('str', {'key': 'bla'}, 'bla'),
        ('np int', {'key': np.int(1)}, np.int64(1)),
        ('np float32', {'key': np.float32(1.1)}, np.float32(1.1)),
        ('np float64', {'key': np.float64(1.1)}, np.float64(1.1)),
        ('np complex64', {'key': np.complex64(1.1j)}, np.complex64(1.1j)),
        ('np complex128', {'key': np.complex128(1.1j)}, np.complex128(1.1j)),
        ('np float64', {'key': np.float64(1.1)}, np.float64(1.1)),
        ('list', {'key': [1, 2, 3]}, [1, 2, 3]),  # Note type is list
        ('tuple', {'key': (1, 2, 3)}, np.array([1, 2, 3])),
        # ('set', {'key': {1, 2, 3}}, {1, 2, 3}),
        ('array', {'key': np.array([1, 2, 3])}, np.array([1, 2, 3])),
        ('np nan', {'key': np.NaN}, np.float64(np.NaN)),
        ('np inf', {'key': np.inf}, np.float64(np.inf)),
        ('np array nan inf', {'key': np.asarray([0, 1, np.nan, np.inf])},
         np.asarray([0, 1, np.nan, np.inf])),
        ('list', {'key': [1.2, [3, 4]]}, [1.2, [3, 4]]),  # Note type is list
    ])
    def test_noop_comma(self, name, data, expect):
        hdf5_dump(data, os.path.join(self.dir, 'test.hdf5'))
        data_load = hdf5_load(os.path.join(self.dir, 'test.hdf5'))

        assert 'key' in data_load.keys(), data_load

        assert type(expect) is type(data_load['key']), \
            (type(expect), type(data_load['key']))
        tc.assert_equal(expect, data_load['key'])
