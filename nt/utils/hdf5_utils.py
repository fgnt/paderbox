import numpy as np
import h5py
import os

from nt.utils import AttrDict

__all__ = ['hdf5_dump']


def hdf5_dump(obj, filename, force=True):
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
    ...    # 'fav_numbers5': [[2], 1], # currently not supported
    ...    'fav_tensors': {
    ...        'levi_civita3d': np.array([
    ...            [[0,0,0],[0,0,1],[0,-1,0]],
    ...            [[0,0,-1],[0,0,0],[1,0,0]],
    ...            [[0,1,0],[-1,0,0],[0,0,0]]
    ...        ]),
    ...        'kronecker2d': np.identity(3)
    ...    }
    ... }
    >>> hdf5_dump(ex, 'tmp_foo.hdf5', True)

    """
    _ReportInterface.__save_dict_to_hdf5__(obj, filename, force=force)


def hdf5_load(filename):
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
    ...    # 'fav_numbers5': [[2], 1], # currently not supported
    ...    'fav_tensors': {
    ...        'levi_civita3d': np.array([
    ...            [[0,0,0],[0,0,1],[0,-1,0]],
    ...            [[0,0,-1],[0,0,0],[1,0,0]],
    ...            [[0,1,0],[-1,0,0],[0,0,0]]
    ...        ]),
    ...        'kronecker2d': np.identity(3)
    ...    }
    ... }
    >>> hdf5_dump(ex, 'tmp_foo.hdf5', True)
    >>> ex_load = hdf5_load('tmp_foo.hdf5', True)
    >>> from pprint import pprint
    >>> ex_load.fav_tensors.kronecker2d[0, 0]
    1.0
    >>> pprint(ex_load)
    {'age': 24,
     'age2': 25,
     'age3': 25j,
     'fav_numbers': array([ 2. ,  4. ,  4.3]),
     'fav_numbers2': array([ 2. ,  4. ,  4.3]),
     'fav_numbers3': array([ 2. ,  4. ,  4.3]),
     'fav_tensors': {'kronecker2d': array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]]),
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
    return _ReportInterface.__load_dict_from_hdf5__(filename)


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
    def __recursively_save_dict_contents_to_group__(cls, h5file, path, dic):
        """..."""
        # argument type checking
        if not isinstance(dic, dict):
            raise ValueError("must provide a dictionary")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        for key, item in dic.items():
            cur_path = os.path.join(path, key)
            if not isinstance(key, str):
                raise ValueError("dict keys must be strings to save to hdf5")
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64,
                                 str, complex, int, float)):
                h5file[cur_path] = item
                if not h5file[cur_path].value == item:
                    raise ValueError('The data representation in the HDF5 '
                                     'file does not match the original dict.')
            # save numpy arrays
            elif isinstance(item, (np.ndarray, list, tuple)):
                h5file[cur_path] = item
                if not np.array_equal(h5file[cur_path].value, item):
                    raise ValueError('The data representation in the HDF5 '
                                     'file does not match the original dict.')
            # save dictionaries
            elif isinstance(item, dict):
                cls.__recursively_save_dict_contents_to_group__(
                    h5file, cur_path, item)
            # other types cannot be saved and will result in an error
            else:
                raise ValueError('Cannot save %s type.' % type(item))

    @classmethod
    def __load_dict_from_hdf5__(cls, filename):
        """..."""
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(
                h5file, '/')

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path):
        """..."""
        ans = AttrDict()
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.__recursively_load_dict_contents_from_group__(
                    h5file, path + key + '/')
        return ans
