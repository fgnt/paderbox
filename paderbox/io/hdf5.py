import io
import os
import warnings

import numpy as np


__all__ = ['dump_hdf5', 'update_hdf5', 'load_hdf5']


def dump_hdf5(obj, filename, force=True, _print_warnings=False):
    """

    ToDo:
        - Discuss default value for force (CB: should be False)

    Args:
        obj:
            HDF5 serializable obj
        filename:
        force:
            IF True skip non HDF5 serializable parts in obj
        _print_warnings:
            Whether to use the warning module or print.
            pytest has some problems with redirect_stderr to catch the
            warnings -> use this option in doctests

    Returns:

    >>> from paderbox.io.cache_dir import get_cache_dir
    >>> from contextlib import redirect_stderr
    >>> import sys
    >>> file = get_cache_dir() / 'tmp.hdf5'
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
    >>> dump_hdf5(ex, file, True, _print_warnings=True)
    >>> ex = {
    ...    'fav_numbers4': {2,4,4.3}, # currently not supported
    ... }
    >>> dump_hdf5(ex, file, True, _print_warnings=True)
    WARNING: Cannot save <class 'set'> type for key "fav_numbers4". Skip this item.
    """
    _ReportInterface.__save_dict_to_hdf5__(
        obj, filename, force=force, print_warnings=_print_warnings)


def update_hdf5(
        obj,
        filename,
        path='/',
        allow_overwrite=False,
        rewrite=False,
):
    """

    ATTENTION: 'allow_overwrite' mean overwrite the index,
               not the data. The data can never be erased.
               Use rewrite==True to delete overwitten data.

    ToDO:
        - Discuss name allow_overwrite
        - Discuss default value for allow_overwrite
        - Discuss error handling when allow_overwrite is False
           - throw an exception
           - print a warning and do not write

    >>> from paderbox.io.cache_dir import get_cache_dir
    >>> from pprint import pprint
    >>> file = get_cache_dir() / 'tmp.hdf5'
    >>> ex = {
    ...    'name': 'stefan',
    ... }
    >>> dump_hdf5(ex, file, True)
    >>> pprint(load_hdf5(file))
    {'name': 'stefan'}
    >>> update_hdf5('peter', file, '/name')
    Traceback (most recent call last):
    ...
    Exception: Path '/name' already exists. Set allow_overwrite to True if you want to overwrite the value. ATTENTION: Overwrite mean overwrite the index, not the data. The data can never be erased.
    >>> update_hdf5('peter', file, '/name', allow_overwrite=True)
    >>> pprint(load_hdf5(file))
    {'name': 'peter'}
    >>> update_hdf5({'name': 1}, file, '/', allow_overwrite=True)
    >>> pprint(load_hdf5(file))
    {'name': 1}
    """
    if not isinstance(obj, dict):
        path_split = os.path.split(path)
        if len(path_split) > 1:
            obj = {path_split[-1]: obj}
            path = os.path.join(*path_split[:-1])
    _ReportInterface.__update_hdf5_from_dict__(
        dic=obj,
        filename=filename,
        path=path,
        allow_overwrite=allow_overwrite
    )
    if rewrite:
        rewrite_hdf5(filename)


def rewrite_hdf5(filename):
    """

    load and dump filename

    The reason for this function is that deleting an element in an hdf5 file
    only deletes the pointer to the data not the data itself. Since the pointer
    is deleted, the data cannot be loaded and consecutively a load dump deletes
    the data.

    """
    dump_hdf5(load_hdf5(filename), filename)


def load_hdf5(filename, path='/'):
    """

    >>> from paderbox.io.cache_dir import get_cache_dir
    >>> file = get_cache_dir() / 'tmp.hdf5'
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
    >>> dump_hdf5(ex, file, True)
    >>> ex_load = load_hdf5(file)
    >>> from pprint import pprint
    >>> ex_load['fav_tensors']['kronecker2d'][0, 0]
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


def tree_hdf5(
        filename,
):
    """

    ToDo:
        Handle list in the same way as load and dump

    >>> from paderbox.io.cache_dir import get_cache_dir
    >>> from IPython.lib.pretty import pprint
    >>> file = get_cache_dir() / 'tmp.hdf5'
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
    >>> dump_hdf5(ex, file, True)
    >>> pprint(tree_hdf5(file))
    ['/age',
     '/age2',
     '/age3',
     '/fav_numbers',
     "/fav_numbers2_<class 'list'>/0",
     "/fav_numbers2_<class 'list'>/1",
     "/fav_numbers2_<class 'list'>/2",
     '/fav_numbers3',
     "/fav_numbers5_<class 'list'>/0_<class 'list'>/0",
     "/fav_numbers5_<class 'list'>/1",
     '/fav_tensors/kronecker2d',
     '/fav_tensors/levi_civita3d',
     '/name']
    """
    import h5py
    path = '/'
    l = []

    def tree(h5file, path):
        for k, v in h5file.items():
            key = os.path.join(path, k)
            if isinstance(v, h5py._hl.group.Group):
                tree(v, key)
            else:
                l.append(key)

    with h5py.File(filename, 'r') as h5file:
        tree(h5file, path)
        return l


class Hdf5DumpWarning(UserWarning):
    pass


# http://codereview.stackexchange.com/a/121314
class _ReportInterface(object):

    @classmethod
    def __save_dict_to_hdf5__(cls, dic, filename, force=False,
                              print_warnings=False):
        """..."""
        import h5py
        if not force and os.path.exists(filename):
            raise ValueError('File %s exists, will not overwrite.' % filename)
        with h5py.File(filename, 'w') as h5file:
            cls.__recursively_save_dict_contents_to_group__(
                h5file,
                '/',
                dic,
                allow_overwrite=False,
                print_warnings=print_warnings,
            )

    @classmethod
    def __update_hdf5_from_dict__(
            cls,
            dic,
            filename,
            path='/',
            allow_overwrite=False,
            print_warnings=False,
    ):
        """..."""
        import h5py
        if isinstance(filename, h5py.File):
            cls.__recursively_save_dict_contents_to_group__(
                filename,
                path,
                dic,
                allow_overwrite=allow_overwrite,
                print_warnings=print_warnings,
            )
        else:
            with h5py.File(filename, 'a') as h5file:
                cls.__recursively_save_dict_contents_to_group__(
                    h5file,
                    path,
                    dic,
                    allow_overwrite=allow_overwrite,
                    print_warnings=print_warnings,
                )

    @classmethod
    def _dump_warning(cls, msg, print_warnings):
        if print_warnings:
            print('WARNING:', msg)
        else:
            warnings.warn(msg, Hdf5DumpWarning, stacklevel=2)

    @classmethod
    def __recursively_save_dict_contents_to_group__(
            cls,
            h5file,
            path,
            dic,
            allow_overwrite,
            print_warnings,
    ):
        """..."""
        import h5py
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
                    f"Skip this item.",
                    print_warnings=print_warnings,
                )
                continue

            def ckeck_exists():
                if cur_path in h5file:
                    if allow_overwrite:
                        del h5file[cur_path]
                    else:
                        raise Exception(
                            f'Path {cur_path!r} already exists. '
                            'Set allow_overwrite to True if you want to '
                            'overwrite the value. '
                            'ATTENTION: Overwrite mean overwrite the index, '
                            'not the data. The data can never be erased.'
                        )

            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64, np.float32,
                                 np.complex64, np.complex128,
                                 str, complex, int, float)):
                ckeck_exists()

                h5file[cur_path] = item

                # NaN compares negative in Python.
                # dataset.value has been deprecated. Use dataset[()] instead.
                if not h5file[cur_path][()] == item and \
                        not np.isnan(item) and \
                        not np.isnan(h5file[cur_path][()]):
                    raise ValueError('The data representation in the HDF5 '
                                     'file does not match the original dict.')
            elif isinstance(item, type(None)):
                ckeck_exists()

                h5file[cur_path] = 'None'
            # save numpy arrays
            elif isinstance(item, (np.ndarray, tuple)):
                ckeck_exists()

                try:
                    h5file[cur_path] = item
                except TypeError as e:
                    cls._dump_warning(
                        f'Cannot save {type(item)} type for key {key}. '
                        f'Error msg: {" ".join(e.args)}. '
                        f'Skip this item.',
                        print_warnings=print_warnings,
                    )
                    continue
                try:
                    # dataset.value has been deprecated. Use dataset[()] instead.
                    np.testing.assert_equal(h5file[cur_path][()], item)
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
                    h5file,
                    cur_path,
                    item,
                    allow_overwrite=allow_overwrite,
                    print_warnings=print_warnings,
                )
            # save lists
            elif isinstance(item, list):
                cls.__recursively_save_dict_contents_to_group__(
                    h5file,
                    cur_path + "_<class 'list'>",
                    {f'{k}': v for k, v in enumerate(item)},
                    allow_overwrite=allow_overwrite,
                    print_warnings=print_warnings,
                )
            # other types cannot be saved and will result in an error
            else:
                cls._dump_warning(
                    f'Cannot save {type(item)} type for key "{key}". '
                    f'Skip this item.',
                    print_warnings=print_warnings,
                )
                continue

    @classmethod
    def __load_dict_from_hdf5__(cls, filename, path='/'):
        """..."""
        import h5py
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(
                h5file, path)

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path):
        """
        >>> from paderbox.io.cache_dir import get_cache_dir
        >>> file = get_cache_dir() / 'tmp.hdf5'
        >>> ex = {'key': [1, 2, 3]}
        >>> dump_hdf5(ex, file, True)
        >>> ex_load = load_hdf5(file)
        >>> ex_load
        {'key': [1, 2, 3]}
        """
        import h5py
        ans = dict()
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
                # dataset.value has been deprecated. Use dataset[()] instead.
                ans[key] = item[()]
                if isinstance(ans[key], str):
                    if ans[key] == 'None':
                        ans[key] = None
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.__recursively_load_dict_contents_from_group__(
                    h5file, path + key + '/')
            else:
                raise TypeError(type(item))
        return ans
