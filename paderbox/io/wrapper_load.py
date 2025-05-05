import os
import io
import gzip
import json
import pickle
import functools
from pathlib import Path
import contextlib

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
            elif hasattr(file, 'name'):
                # A file descriptor can have a name attribute, with the path
                # that it represents. Use this information to get the suffix.
                # Note: For BytesIO and StringIO, you can manually add the name
                #       attribute, see:
                #       https://stackoverflow.com/a/42811024/5766934
                ext = Path(file.name).suffix
            else:
                raise ValueError(
                    f'{type(file)} is not supported without the argument "ext.\n"'
                    f'e.g. load(..., ext=".json")'
                )
            # load_audio uses ".../file.wav::[slice]" notation to load a part 
            # of an audio file. We have to remove the slice form the extension 
            # for a correct dispatch
            ext = ext.split('::')[0]
        else:
            ext = self.ext

        if ext in ['.json']:
            with file.open('r') as fp:
                return json.load(fp, **self.kwargs)
        elif ext in ['.pkl', '.dill']:
            assert self.unsafe, self._unsafe_msg(self.unsafe, file, ext)
            with file.open('rb') as fp:
                return pickle.load(fp, **self.kwargs)
        elif ext in ['.h5', '.hdf5']:
            # ToDo: Is hdf5 safe or unsafe?
            from paderbox.io.hdf5 import load_hdf5
            return load_hdf5(file)
        elif ext in ['.yaml', '.yml']:
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
        elif ext in ['.wav', '.flac']:
            from paderbox.io import load_audio
            return load_audio(file, **self.kwargs)
        elif ext in ['.npz']:
            import numpy as np
            data = np.load(file, **self.kwargs, allow_pickle=self.unsafe)
            ret_data = dict(data)
            data.close()
            return ret_data
        elif ext in ['.npy']:
            assert self.unsafe, self._unsafe_msg(self.unsafe, file, ext)
            import numpy as np
            return np.load(file, **self.kwargs, allow_pickle=self.unsafe)
        elif ext in ['.gz', '.json.gz', '.pkl.gz', '.npy.gz', '.wav.gz']:
            if ext == '.gz':
                ext = ''.join(file.suffixes[-2:])

            if isinstance(file, io.IOBase):
                gzip_file = dict(fileobj=file)
            else:
                gzip_file = dict(filename=file)

            with gzip.GzipFile(**gzip_file, mode='rb') as f:
                if ext == '.json.gz':
                    return json.loads(f.read().decode(), **self.kwargs)
                elif ext == '.pkl.gz':
                    assert self.unsafe, self._unsafe_msg(self.unsafe, file, ext)
                    return pickle.load(f, **self.kwargs)
                elif ext == '.npy.gz':
                    assert self.unsafe, (self.unsafe, file)
                    return np.load(f, allow_pickle=self.unsafe, **self.kwargs)
                elif ext == '.wav.gz':
                    from paderbox.io import load_audio
                    return load_audio(f, **self.kwargs)
                else:
                    raise ValueError(ext, file)
        elif ext in ['.wv1', '.wv2']:
            from paderbox.io.audioread import read_nist_wsj
            date, sampling_rate = read_nist_wsj(file)
            return date
        elif ext in ['.pth']:
            return self.pth(file, map_location='cpu', weights_only=not self.unsafe)
        elif ext in ['.mat']:
            # ToDo: Is hdf5 safe or unsafe?  (loadmat uses hdf5)
            import scipy.io as sio
            return sio.loadmat(file)
        else:
            if self.ignore_type_error and '.' not in str(file):
                return str(file)
            else:
                raise ValueError(file, ext)

    def pth(self, file, map_location='cpu', **kwargs):
        """
        >>> from packaging import version
        >>> import tempfile
        >>> import torch
        >>> import numpy as np
        >>> loader = Loader(False, None, unsafe=True)
        >>> with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        ...     torch.save({'a': os.system}, tmp.name)
        ...     loader(tmp.name)
        {'a': <built-in function system>}

        >>> loader = Loader(False, None, unsafe=False)
        >>> with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        ...     try:
        ...         torch.save({'a': os.system}, tmp.name)
        ...         loader(tmp.name)
        ...     except Exception as e:
        ...         print("Failed as it should. The message and type changes between torch versions.")
        Failed as it should. The message and type changes between torch versions.

            
        >>> if  version.parse(torch.__version__) < version.parse('2.6'):
        ...     import pytest
        ...     pytest.skip("Proper weights_only is only supported in torch >= 2.6")

        >>> with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        ...     torch.save({'a': np.array([1])}, tmp.name)
        ...     loader(tmp.name)
        {'a': array([1])}
        """
        kwargs['map_location'] = map_location

        from packaging import version  # distutils is deprecated
        import torch

        weights_only = kwargs.pop('weights_only', not self.unsafe)
        
        if version.parse(torch.__version__) < version.parse('2.6'):
            assert not weights_only, self._unsafe_msg(f'{self.unsafe} (weights_only={weights_only})', file, '.pth')
            return torch.load(str(file), **kwargs)
        elif not weights_only:
            return torch.load(str(file), weights_only=False, **kwargs)
        else:
            import numpy as np

            # Original code for _safe_globals:
            #  - https://github.com/huggingface/transformers/pull/34632
            #  - https://github.com/fgnt/padertorch/pull/163
            def _safe_globals():
                # Starting from version 2.4 PyTorch introduces a check for the objects
                # loaded with torch.load(weights_only=True). Starting from 2.6
                # weights_only=True becomes a default and requires allowlisting of objects
                # being loaded.
                # CB: torch.serialization.safe_globals was introduced in 2.5 or 2.6.
                # See: https://github.com/pytorch/pytorch/pull/137602
                # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
                # See: https://github.com/huggingface/accelerate/pull/3036

                np_core = (
                    np._core if version.parse(np.__version__)
                    >= version.parse("2.0.0") else np.core
                )
                allowlist = [
                    np_core.multiarray._reconstruct,  # added in transformers/pull/34632
                    np_core.multiarray.scalar,  # added in padertorch/pull/163
                    np.ndarray,  # added in transformers/pull/34632
                    np.dtype,  # added in transformers/pull/34632
                    # numpy >1.25 defines numpy.dtypes.UInt32DType, but type works for all versions of numpy
                    type(np.dtype(np.int64)),  # added in initial paderbox commit
                    type(np.dtype(np.uint32)),  # added in transformers/pull/34632
                    type(np.dtype(np.float64)),  # added in padertorch/pull/163
                    type(np.dtype(np.float32)),  # added in initial paderbox commit
                ]
                
                return torch.serialization.safe_globals(allowlist)
            
            with _safe_globals():
                return torch.load(str(file), weights_only=True, **kwargs)

    def _unsafe_msg(self, unsafe, file, ext):
        return (
            f'You called {self.__call__.__qualname__} with unsafe={unsafe}\n'
            f'for the file {file}.\n'
            f'The file type is identified as {ext!r}.\n'
            f'Loading this file type is not secure.\n'
            f'If you trust the file, change the value of unsafe to True.\n'
            f'\n'
            f'From https://docs.python.org/3/library/pickle.html:\n'
            f'  It is possible to construct malicious pickle data which will\n'
            f'  execute arbitrary code during unpickling. Never unpickle\n'
            f'  data that could have come from an untrusted source, or that\n'
            f'  could have been tampered with.'
        )

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
            Flag, to indicate, if you want to allow the loading from
            unsecure files, e.g. pickle.
        **kwargs:
            kwargs for the specific loader.

    Returns:


    To load an unsecure, you have to change `unsafe` to `True`
    >>> load('/path/to/unsecure_file.pkl', unsafe=False)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AssertionError: You called Loader.__call__ with unsafe=False
    for the file ...unsecure_file.pkl.
    The file type is identified as '.pkl'.
    Loading this file type is not secure.
    If you trust the file, change the value of unsafe to True.
    <BLANKLINE>
    From https://docs.python.org/3/library/pickle.html:
      It is possible to construct malicious pickle data which will
      execute arbitrary code during unpickling. Never unpickle
      data that could have come from an untrusted source, or that
      could have been tampered with.

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
