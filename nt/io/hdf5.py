import io
import os
import warnings

import h5py
import numpy as np
from nt.utils import AttrDict

__all__ = ['dump_hdf5', 'update_hdf5', 'load_hdf5']


def dump_hdf5(obj, filename, force=True):
    """

    >>> from contextlib import redirect_stderr
    >>> import sys
    >>> ex = {
    ...    'name': 'stefan',
    ...    'age':  np.int64(24),
    ...    'age2':  25,
    ...    'age3':  25j,
    ...    'age4':  np.float32(24.2),
    ...    'age5':  np.float64(24.2),
    ...    'fav_numbers': np.array([2,4,4.3]),
    ...    'fav_numbers2': [2,4,4.3],
    ...    'fav_numbers3': (2,4,4.3),
    ...    # 'fav_numbers4': {2,4,4.3}, # currently not supported
    ...    'fav_numbers5': [[2], 1],
    ...    'fav_tensors': {
    ...        'levi_civita3d': np.array([
    ...            [[0,0,0],[0,0,1],[0,-1,0]],
    ...            [[0,0,-1],[0,0,0],[1,0,0]],
    ...            [[0,1,0],[-1,0,0],[0,0,0]]
    ...        ]),
    ...        'kronecker2d': np.identity(3)
    ...    }
    ... }
    >>> with redirect_stderr(sys.stdout):
    ...     dump_hdf5(ex, 'tmp_foo.hdf5', True)
    >>> ex = {
    ...    'fav_numbers4': {2,4,4.3}, # currently not supported
    ... }
    >>> with io.StringIO() as buf, redirect_stderr(buf):
    ...     dump_hdf5(ex, 'tmp_foo.hdf5', True)
    ...     s = buf.getvalue()
    ...     assert 'Hdf5DumpWarning' in s
    ...     assert 'fav_numbers4' in s
    """
    _ReportInterface.__save_dict_to_hdf5__(obj, filename, force=force)


def update_hdf5(obj, filename, path='/'):
    """
    """
    _ReportInterface.__update_hdf5_from_dict__(obj, filename, path)


def load_hdf5(filename, path='/'):
    """

    >>> ex = {
    ...    'name': 'stefan',
    ...    'age':  np.int64(24),
    ...    'age2':  25,
    ...    'age3':  25j,
    ...    'fav_numbers': np.array([2,4,4.3]),
    ...    'fav_numbers2': [2,4,4.3],
    ...    'fav_numbers3': (2,4,4.3),
    ...    # 'fav_numbers4': {2,4,4.3}, # currently not supported
    ...    'fav_numbers5': [[2], 1],
    ...    'fav_tensors': {
    ...        'levi_civita3d': np.array([
    ...            [[0,0,0],[0,0,1],[0,-1,0]],
    ...            [[0,0,-1],[0,0,0],[1,0,0]],
    ...            [[0,1,0],[-1,0,0],[0,0,0]]
    ...        ]),
    ...        'kronecker2d': np.identity(3)
    ...    }
    ... }
    >>> dump_hdf5(ex, 'tmp_foo.hdf5', True)
    >>> ex_load = load_hdf5('tmp_foo.hdf5')
    >>> from pprint import pprint
    >>> ex_load.fav_tensors.kronecker2d[0, 0]
    1.0
    >>> pprint(ex_load)
    {'age': 24,
     'age2': 25,
     'age3': 25j,
     'fav_numbers': array([2. , 4. , 4.3]),
     'fav_numbers2': [2, 4, 4.3],
     'fav_numbers3': array([2. , 4. , 4.3]),
     'fav_numbers5': [[2], 1],
     'fav_tensors': {'kronecker2d': array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]),
                     'levi_civita3d': array([[[ 0,  0,  0],
            [ 0,  0,  1],
            [ 0, -1,  0]],
    <BLANKLINE>
           [[ 0,  0, -1],
            [ 0,  0,  0],
            [ 1,  0,  0]],
    <BLANKLINE>
           [[ 0,  1,  0],
            [-1,  0,  0],
            [ 0,  0,  0]]])},
     'name': 'stefan'}
    """
    return _ReportInterface.__load_dict_from_hdf5__(filename, path=path)


class Hdf5DumpWarning(UserWarning):
    pass


# http://codereview.stackexchange.com/a/121314
class _ReportInterface(object):

    @classmethod
    def __save_dict_to_hdf5__(cls, dic, filename, force=False):
        """..."""
        if not force and os.path.exists(filename):
            raise ValueError('File %s exists, will not overwrite.' % filename)
        with h5py.File(filename, 'w') as h5file:
            cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic)

    @classmethod
    def __update_hdf5_from_dict__(cls, dic, filename, path='/'):
        """..."""
        if isinstance(filename, h5py.File):
            cls.__recursively_save_dict_contents_to_group__(filename, path, dic)
        else:
            with h5py.File(filename, 'a') as h5file:
                cls.__recursively_save_dict_contents_to_group__(h5file, path, dic)

    @classmethod
    def _dump_warning(cls, msg):
        warnings.warn(msg, Hdf5DumpWarning, stacklevel=2)

    @classmethod
    def __recursively_save_dict_contents_to_group__(cls, h5file, path, dic):
        """..."""
        # argument type checking
        if not isinstance(dic, dict):
            raise ValueError(f"must provide a dictionary, not "
                             f"{type(dict)}")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            # TODO: This is true for closed h5 files, too.
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        for key, item in dic.items():
            cur_path = os.path.join(path, key)
            if not isinstance(key, str):
                cls._dump_warning(
                    f"dict keys must be strings (and not {key}) "
                    f"to save to hdf5. "
                    f"Skip this item."
                )
                continue
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64, np.float32,
                                 np.complex64, np.complex128,
                                 str, complex, int, float)):
                h5file[cur_path] = item

                # NaN compares negative in Python.
                if not h5file[cur_path].value == item and \
                        not np.isnan(item) and \
                        not np.isnan(h5file[cur_path].value):
                    raise ValueError('The data representation in the HDF5 '
                                     'file does not match the original dict.')
            elif isinstance(item, type(None)):
                h5file[cur_path] = 'None'
            # save numpy arrays
            elif isinstance(item, (np.ndarray, tuple)):
                try:
                    h5file[cur_path] = item
                except TypeError as e:
                    cls._dump_warning(
                        f'Cannot save {type(item)} type for key {key}. '
                        f'Error msg: {" ".join(e.args)}. '
                        f'Skip this item.'
                    )
                    continue
                try:
                    np.testing.assert_equal(h5file[cur_path].value, item)
                except AssertionError:
                    raise ValueError('The data representation in the HDF5 '
                                     'file does not match the original dict.')
            elif isinstance(item, h5py.SoftLink):
                # allow SoftLink to be overwritten
                if cur_path in h5file.keys():
                    del h5file[cur_path]
                h5file[cur_path] = item
            # save dictionaries
            elif isinstance(item, dict):
                cls.__recursively_save_dict_contents_to_group__(
                    h5file, cur_path, item)
            # save lists
            elif isinstance(item, list):
                cls.__recursively_save_dict_contents_to_group__(
                    h5file, cur_path + "_<class 'list'>",
                    {f'{k}': v for k, v in enumerate(item)}
                )
            # other types cannot be saved and will result in an error
            else:
                cls._dump_warning(
                    f'Cannot save {type(item)} type for key "{key}". '
                    f'Skip this item.'
                )
                continue

    @classmethod
    def __load_dict_from_hdf5__(cls, filename, path='/'):
        """..."""
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(
                h5file, path)

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path):
        """

        >>> ex = {'key': [1, 2, 3]}
        >>> dump_hdf5(ex, 'tmp_foo.hdf5', True)
        >>> ex_load = load_hdf5('tmp_foo.hdf5')
        >>> ex_load
        {'key': [1, 2, 3]}
        """
        ans = AttrDict()
        for key, item in h5file[path].items():
            if key.endswith("_<class 'list'>"):
                tmp = cls.__recursively_load_dict_contents_from_group__(
                    h5file, path + key + '/')
                ans[key[:-len("_<class 'list'>")]] = \
                    [value for (key, value) in sorted(tmp.items(),
                                                      key=lambda x: int(x[0]))]
            # Support old format
            elif key.endswith("_list"):
                tmp = cls.__recursively_load_dict_contents_from_group__(
                    h5file, path + key + '/')
                ans[key.rstrip("_list")] = \
                    [value for (key, value) in sorted(tmp.items(),
                                                      key=lambda x: int(x[0]))]
            elif isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
                if isinstance(ans[key], str):
                    if ans[key] == 'None':
                        ans[key] = None
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.__recursively_load_dict_contents_from_group__(
                    h5file, path + key + '/')
            else:
                raise TypeError(type(item))
        return ans
