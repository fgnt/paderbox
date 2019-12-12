import gzip
import json
import pickle

import numpy as np

# np, yaml and soundfile are slow imports, make them lazy

from paderbox.io.path_utils import normalize_path

__all__ = ['dump']


def dump(
        obj,
        path,
        mkdir=False,
        mkdir_parents=False,
        mkdir_exist_ok=False,  # Should this be an option? Should the default be True?
        unsafe=False,  # Should this be an option? Should the default be True?
        # atomic=False,  ToDo: Add atomic support
        **kwargs,
):
    """
    A generic dump function to write the obj to path.

    Infer the dump protocol (e.g. json, pickle, ...) from the path name.

    Supported formats:
     - Text:
       - json
       - yaml
     - Binary:
       - pkl: pickle
       - dill
       - h5: HDF5
       - wav
       - mat: MATLAB
       - npy: Numpy
       - npz: Numpy compressed
       - pth: Pickle with Pytorch support
     - Compressed:
       - json.gz
       - pkl.gz
       - npy.gz

    Args:
        obj: Arbitrary object that is supported from the dump protocol.
        path: str or pathlib.Path
        mkdir:
            Whether to make an mkdir id the parent dir of path does not exist.
        mkdir_parents:
        mkdir_exist_ok:
        unsafe:
            Allow unsafe dump protocol. This option is more relevant for load.
        **kwargs:
            Forwarded arguments to the particular dump function.
            Should rarely be used, because when a special property of the dump
            function/protocol is used, use directly that dump function.

    Returns:

    """
    path = normalize_path(path, allow_fd=False)
    if mkdir:
        if mkdir_exist_ok:
            # Assume that in most cases the dir exists.
            # -> try first to reduce io requests
            try:
                return dump(obj, path, unsafe=unsafe, **kwargs)
            except FileNotFoundError:
                pass
        path.parent.mkdir(parents=mkdir_parents, exist_ok=mkdir_exist_ok)

    if str(path).endswith(".json"):
        from paderbox.io import dump_json
        dump_json(obj, path, **kwargs)
    elif str(path).endswith(".pkl"):
        assert unsafe, (unsafe, path)
        with path.open("wb") as fp:
            pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL, **kwargs)
    elif str(path).endswith(".dill"):
        assert unsafe, (unsafe, path)
        with path.open("wb") as fp:
            import dill
            dill.dump(obj, fp, **kwargs)
    elif str(path).endswith(".h5"):
        from paderbox.io.hdf5 import dump_hdf5
        dump_hdf5(obj, path, **kwargs)
    elif str(path).endswith(".yaml"):
        if unsafe:
            from paderbox.io.yaml_module import dump_yaml_unsafe
            dump_yaml_unsafe(obj, path, **kwargs)
        else:
            from paderbox.io.yaml_module import dump_yaml
            dump_yaml(obj, path, **kwargs)
    elif str(path).endswith(".gz"):
        assert len(kwargs) == 0, kwargs
        with gzip.GzipFile(path, 'wb', compresslevel=1) as f:
            if str(path).endswith(".json.gz"):
                f.write(json.dumps(obj).encode())
            elif str(path).endswith(".pkl.gz"):
                assert unsafe, (unsafe, path)
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif str(path).endswith(".npy.gz"):
                np.save(f, obj, allow_pickle=unsafe)
            else:
                raise ValueError(path)
    elif str(path).endswith(".wav"):
        from paderbox.io import dump_audio
        if np.ndim(obj) == 1:
            pass
        elif np.ndim(obj) == 2:
            assert np.shape(obj)[0] < 20, (np.shape(obj), obj)
        else:
            raise AssertionError(('Expect ndim in [1, 2]', np.shape(obj), obj))
        with path.open("wb") as fp:  # Throws better exception msg
            dump_audio(obj, fp, **kwargs)
    elif str(path).endswith('.mat'):
        import scipy.io as sio
        sio.savemat(path, obj, **kwargs)
    elif str(path).endswith('.npy'):
        np.save(str(path), obj, allow_pickle=unsafe, **kwargs)
    elif str(path).endswith('.npz'):
        assert unsafe, (unsafe, path)
        assert len(kwargs) == 0, kwargs
        if isinstance(obj, dict):
            np.savez(str(path), **obj)
        else:
            np.savez(str(path), obj)
    elif str(path).endswith('.pth'):
        assert unsafe, (unsafe, path)
        import torch
        torch.save(obj, str(path), **kwargs)
    else:
        raise ValueError('Unsupported suffix:', path)
