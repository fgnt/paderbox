import io
from pathlib import Path


def normalize_path(file, as_str=False, allow_fd=True):
    if allow_fd and isinstance(file, io.IOBase):
        return file
    elif isinstance(file, (str, Path)):
        # expanduser: ~ to username
        # resolve: better exception i.e. absolute path
        file = Path(file).expanduser().resolve()
        if as_str:
            return str(file)
        else:
            return file
    else:
        raise TypeError(file)
