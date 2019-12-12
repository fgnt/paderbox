import io
import gzip
import json
import pickle
import functools
from pathlib import Path

from paderbox.io.path_utils import normalize_path


class Loader:
    """
    Loads a single file.

    Args:
        file:
        ignore_type_error:
        ext:
        unsafe:
        **kwargs:

    Returns:

    """
    def __init__(self, ignore_type_error, ext, unsafe, **kwargs):
        self.ignore_type_error = ignore_type_error
        self.ext = ext
        self.unsafe = unsafe
        self.kwargs = kwargs

    def __call__(self, file):
        import numpy as np
        file = normalize_path(file)

        if self.ext is None:
            if isinstance(file, (str, Path)):
                file = Path(file)
                ext = file.suffix
            else:
                raise ValueError(
                    f'{type(file)} is not supported without the argument "ext.\n"'
                    f'e.g. load(..., ext=".json")'
                )
        else:
            ext = self.ext

        if ext in ['.json']:
            with file.open('r') as fp:
                return json.load(fp, **self.kwargs)
        elif ext in ['.pkl', '.dill']:
            assert self.unsafe, (self.unsafe, file)
            with file.open('rb') as fp:
                return pickle.load(fp, **self.kwargs)
        elif ext in ['.h5']:
            # ToDo: Is hdf5 safe or unsafe?
            from paderbox.io.hdf5 import load_hdf5
            return load_hdf5(file)
        elif ext in ['.yaml']:
            with file.open('r') as fp:
                import yaml
                if self.unsafe:
                    return yaml.unsafe_load(fp, **self.kwargs)
                else:
                    return yaml.safe_load(fp, **self.kwargs)
        # elif ext in ['yaml_all']:
        #     with file.open('r') as fp:
        #         import yaml
        #         return yaml.load_all(fp, **kwargs)
        elif ext in ['.wav']:
            from paderbox.io import load_audio
            return load_audio(file, **self.kwargs)
        elif ext in ['.npz']:
            import numpy as np
            data = np.load(file, **self.kwargs, allow_pickle=self.unsafe)
            ret_data = dict(data)
            data.close()
            return ret_data
        elif ext in ['.npy']:
            assert self.unsafe, (self.unsafe, file)
            import numpy as np
            return np.load(file, **self.kwargs, allow_pickle=self.unsafe)
        elif ext in ['.gz', '.json.gz', '.pkl.gz', '.npy.gz']:
            with gzip.GzipFile(file, 'rb') as f:
                if str(file).endswith('.json.gz'):
                    return json.loads(f.read().decode())
                elif str(file).endswith('.pkl.gz'):
                    assert self.unsafe, (self.unsafe, file)
                    return pickle.load(f)
                elif str(file).endswith('.npy.gz'):
                    assert self.unsafe, (self.unsafe, file)
                    return np.load(f, allow_pickle=self.unsafe)
                else:
                    raise ValueError(file)
        elif ext in ['.wv1', '.wv2']:
            from paderbox.io.audioread import read_nist_wsj
            date, sampling_rate = read_nist_wsj(file)
            return date
        elif ext in ['.pth']:
            assert self.unsafe, (self.unsafe, file)
            import torch
            return torch.load(str(file), map_location='cpu')
        elif str(file).endswith('.mat'):
            # ToDo: Is hdf5 safe or unsafe?  (loadmat uses hdf5)
            import scipy.io as sio
            return sio.loadmat(file)
        else:
            if self.ignore_type_error and '.' not in str(file):
                return str(file)
            else:
                raise ValueError(file)


def recursive_load(
        obj,
        *,
        loader,
        list_to='dict',
        ignore_type_error=False,
):
    """
    Args:
        obj:
            A nested object of dict, list and tuple.
            The leafs can have the datatypes: `str', Path and/or file
            descriptor.
            Other types are only supported, when ignore_type_error is True.
            These types will be ignored (i.e. not loaded).
        loader:
            Callable object that takes a str, Path or file descriptor as input
            and returns the content of the obj.
        list_to:
        ignore_type_error:
        Whether to ignore

    Returns:

    """
    import numpy as np

    self_call = functools.partial(
        recursive_load,
        loader=loader,
        list_to=list_to,
        ignore_type_error=ignore_type_error,
    )

    if isinstance(obj, (io.IOBase, str, Path)):
        return loader(obj)
    elif isinstance(obj, (list, tuple)):
        if list_to == 'dict':
            return {f: self_call(f) for f in obj}
        elif list_to == 'array':
            return np.array([self_call(f) for f in obj])
        elif list_to == 'list':
            return [self_call(f) for f in obj]
        else:
            raise ValueError(list_to)
    elif isinstance(obj, (dict,)):
        return obj.__class__({k: self_call(v) for k, v in obj.items()})
    else:
        if ignore_type_error:
            return obj
        else:
            raise TypeError(obj)


def load(
        obj,
        *,
        list_to='list',
        ext=None,
        ignore_type_error=False,
        unsafe=False,
        **kwargs,
):
    """
    Loads a nested object.
    To allow unsafe reading (e.g. `pickle`) use `load_unsafe` or set `unsafe`
    to `True`.

    Args:
        obj:
            A nested object of dict, list and tuple.
            The leafs can have the datatypes: `str', Path and/or file
            descriptor.
            Other types are only supported, when ignore_type_error is True.
            These types will be ignored (i.e. not loaded).
        list_to: 'dict', 'list', 'array'
            Example: ['1.json', '2.json'], where '1.json' contains only a 1.
            'dict': {'1.json': 1, '2.json': 1}
            'list': [1, 2]
            'array': np.array([1, 2])
        ext:
            - None: Obtained from obj path (default)
            - str of obj extension, i.e.:
                - Safe obj extensions:
                    - .json
                    - .h5: HDF5 file
                    - .wav: Audio file
                    - .wv1, .wv2: Nist file
                    - .json.gz Compressed json
                - Optional safe/unsafe extensions:
                    - .yaml
                    - .npz, .npy: Numpy file
                - Unsafe extensions:
                    - .pkl, .dill
                    - .pth: Torch file
                    - .pkl.gz: Compressed Pickle
                - Can be used to limit the loader to only support one ext.
            - callable that takes a path as input
        ignore_type_error:
        unsafe:
        **kwargs:
            kwargs for the specific loader.

    Returns:

    """
    loader = Loader(
        ignore_type_error=ignore_type_error,
        ext=ext,
        unsafe=unsafe,
        **kwargs,
    )
    return recursive_load(
        obj,
        loader=loader,
        list_to=list_to,
        ignore_type_error=ignore_type_error,
    )


def load_unsafe(
        obj,
        *,
        list_to='dict',
        ignore_type_error=False,
        ext=None,
        **kwargs,
):
    """
    unsafe: Whether to allow unsafe reading (e.g. allow pickle).

    """
    return load(
        obj,
        ext=ext,
        list_to=list_to,
        ignore_type_error=ignore_type_error,
        **kwargs,
    )
