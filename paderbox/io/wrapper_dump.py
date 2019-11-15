import gzip
import json
import pickle

import numpy as np

# np, yaml and soundfile are slow imports, make them lazy

from paderbox.io.path_utils import normalize_path

__all__ = ['dump']


def dump(
        obj,
        file,
        mkdir=False,
        mkdir_parents=False,
        mkdir_exist_ok=False,  # Should this be an option? Should the default be True?
        unsafe=False,  # Should this be an option? Should the default be True?
        # atomic=False,  ToDo: Add atomic support
        **kwargs,
):
    """
    A generic dump function to write the obj to file.

    Infer the dump protocol (e.g. json, pickle, ...) from the file name.

    Supported formats:
     - Text:
       - json
       - yaml
     - Binary:
       - pickle
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
        file: str or pathlib.Path
        mkdir:
            Whether to make an mkdir id the parent dir of file does not exist.
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
    file = normalize_path(file, allow_fd=False)
    if mkdir:
        if mkdir_exist_ok:
            # Assume that in most cases the dir exists.
            # -> try first to reduce io requests
            try:
                return dump(obj, file, unsafe=unsafe, **kwargs)
            except FileNotFoundError:
                pass
        file.parent.mkdir(parents=mkdir_parents, exist_ok=mkdir_exist_ok)

    if str(file).endswith(".json"):
        from paderbox.io import dump_json
        dump_json(obj, file, **kwargs)
    elif str(file).endswith(".pkl"):
        assert unsafe, (unsafe, file)
        with file.open("wb") as fp:
            pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL, **kwargs)
    elif str(file).endswith(".dill"):
        assert unsafe, (unsafe, file)
        with file.open("wb") as fp:
            import dill
            dill.dump(obj, fp, **kwargs)
    elif str(file).endswith(".h5"):
        from paderbox.io.hdf5 import dump_hdf5
        dump_hdf5(obj, file, **kwargs)
    elif str(file).endswith(".yaml"):
        if unsafe:
            from paderbox.io.yaml_module import dump_yaml_unsafe
            dump_yaml_unsafe(obj, file, **kwargs)
        else:
            from paderbox.io.yaml_module import dump_yaml
            dump_yaml(obj, file, **kwargs)
    elif str(file).endswith(".gz"):
        assert len(kwargs) == 0, kwargs
        with gzip.GzipFile(file, 'wb', compresslevel=1) as f:
            if str(file).endswith(".json.gz"):
                f.write(json.dumps(obj).encode())
            elif str(file).endswith(".pkl.gz"):
                assert unsafe, (unsafe, file)
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif str(file).endswith(".npy.gz"):
                np.save(f, obj, allow_pickle=unsafe)
            else:
                raise ValueError(file)
    elif str(file).endswith(".wav"):
        from paderbox.io import dump_audio
        if np.ndim(obj) == 1:
            pass
        elif np.ndim(obj) == 2:
            assert np.shape(obj)[0] < 20, (np.shape(obj), obj)
        else:
            raise AssertionError(('Expect ndim in [1, 2]', np.shape(obj), obj))
        with file.open("wb") as fp:  # Throws better exception msg
            dump_audio(obj, fp, **kwargs)
    elif str(file).endswith('.mat'):
        import scipy.io as sio
        sio.savemat(file, obj, **kwargs)
    elif str(file).endswith('.npy'):
        np.save(str(file), obj, allow_pickle=unsafe, **kwargs)
    elif str(file).endswith('.npz'):
        assert unsafe, (unsafe, file)
        assert len(kwargs) == 0, kwargs
        if isinstance(obj, dict):
            np.savez(str(file), **obj)
        else:
            np.savez(str(file), obj)
    elif str(file).endswith('.pth'):
        assert unsafe, (unsafe, file)
        import torch
        torch.save(obj, str(file), **kwargs)
    else:
        raise ValueError(file)
