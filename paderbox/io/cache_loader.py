import dataclasses
import pathlib
from pathlib import Path
import collections
import io
import os
import contextlib
import functools

import paderbox as pb
from paderbox.io.atomic import write_bytes_atomic


def check(cache_dir, keepfree_gigabytes=5):
    # copied from lazy_dataset.core.DiskCacheDataset
    import shutil
    diskusage = shutil.disk_usage(cache_dir)
    if diskusage.free < keepfree_gigabytes * 1024 ** 3:
        import warnings
        import humanfriendly
        warnings.warn(
            f'There is not much space left in the specified cache '
            f'dir "{cache_dir}"! (total='
            f'{humanfriendly.format_size(diskusage.total, binary=True)}'
            f', free='
            f'{humanfriendly.format_size(diskusage.free, binary=True)}'
            f')', ResourceWarning
        )
        if diskusage.free < 1 * 1024 ** 3:
            # Crash if less than 1GB left. It's better to crash
            # this process than to crash the whole machine
            raise RuntimeError(
                f'Not enough space on device! The device that the '
                f'cache directory "{cache_dir}" '
                f'is located on has less than 1GB '
                f'space left. You probably want to delete some '
                f'files before crashing the machine.'
            )
    return True


@dataclasses.dataclass
class DiskCacheLoader:
    """
    A file loader that uses a cache dir to speedup the second read.

    Assumption:
        The files that should be read are on a slow filesystem (e.g. network).
        The cache dir is on a fast filesystem (e.g. local disk).

    Note:
        Be careful, that it is not guaranteed, that the files are afterwards
        deleted, because some errors don't trigger the shutdown of the python
        process and simply exit the program. e.g. pressing Ctrl-C multiple
        times may leave some artefacts on the filesystem.


    >>> import tempfile
    >>> from unittest import mock
    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> file = get_file_path('sample.wav')
    >>> file2 = get_file_path('speech.wav')
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     # Use _StatisticsDiskCacheLoader instead of DiskCacheLoader
    ...     # here, to show what is done.
    ...     loader = _StatisticsDiskCacheLoader(cache_dir=tmpdir)
    ...     tmpdir = Path(tmpdir)
    ...     print(loader.statistics)
    ...     _ = loader(file)
    ...     print(loader.statistics)
    ...     _ = loader(file2)
    ...     print(loader.statistics)
    ...     _ = loader(file)
    ...     print(loader.statistics)
    ...     _ = loader(file2)
    ...     print(loader.statistics)
    ...     with loader.open(file2, 'rb') as fd:
    ...         _ = fd.read()
    ...     print(loader.statistics)
    Counter()
    Counter({'buffer calls': 1, 'check enough disk space': 1, 'copy calls': 1, 'coppied .wav': 1})
    Counter({'buffer calls': 2, 'check enough disk space': 2, 'copy calls': 2, 'coppied .wav': 2})
    Counter({'buffer calls': 3, 'check enough disk space': 2, 'copy calls': 2, 'coppied .wav': 2})
    Counter({'buffer calls': 4, 'check enough disk space': 2, 'copy calls': 2, 'coppied .wav': 2})
    Counter({'buffer calls': 5, 'check enough disk space': 2, 'copy calls': 2, 'coppied .wav': 2, 'open calls': 1, 'opened .wav': 1})
    """
    cache_dir: [str, Path]

    mapping: dict = dataclasses.field(default_factory=dict, init=False, repr=False)

    # True: Cleanup the cache_dir and assume it is empty on start.
    clear: bool = True

    keepfree_gigabytes: int = 5

    # Set to False to disable the cache.
    use_cache: bool = True

    # Once the cache is "full" (according to 'keepfree_gigabytes'), write no
    # more files to the cache.
    cache_full: dict = dataclasses.field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        # os.scandir is fast
        if self.clear and len(list(os.scandir(self.cache_dir))) != 0:
            raise RuntimeError(f'Dir {self.cache_dir} is not empty.')

        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                print(f'WARNING: Tried to use {self.cache_dir} as cache, but got a permission error: {e}')
                print('Disable the cache.')
                self.use_cache = False

    def check(self, cache_dir):
        return check(cache_dir, keepfree_gigabytes=self.keepfree_gigabytes)

    def copy(self, old_path, new_path):
        with old_path.open(mode='rb') as f:
            raw = f.read()
        try:
            write_bytes_atomic(raw, new_path)
        except FileNotFoundError:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(raw, new_path)
        except OSError as e:
            if os.name == 'nt' and e.errno == 22:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                write_bytes_atomic(raw, new_path)
            else:
                raise

        return raw

    @staticmethod
    def _get_new_path(cache_dir, path):
        """

        Args:
            cache_dir:
            path: pathlib.Path(...).absolute()

        >>> from pathlib import PureWindowsPath, PurePosixPath
        >>> l = DiskCacheLoader
        >>> l._get_new_path(PurePosixPath('/a/b/c'), PurePosixPath('/d/s'))
        PurePosixPath('/a/b/c/d/s')
        >>> l._get_new_path(PureWindowsPath(r'C:\\\\a\\\\b\\\\e\\\\'), PureWindowsPath(r'D:\\\\f\\\\g'))
        PureWindowsPath('C:/a/b/e/D/f/g')
        """
        # new_path = Path(f'{cache_dir}/{path}')  # if PurePosixPath
        new_path = cache_dir.joinpath(
            path.drive.replace(':', ''),
            path.relative_to(list(path.parents)[-1]))
        return new_path

    def buffer(self, path):
        if not self.use_cache:
            return path, None
        path = Path(path).absolute()
        if path in self.mapping:
            return self.mapping[path], None
        elif self.cache_full:
            return path, None
        else:
            new_path = self._get_new_path(self.cache_dir, path)
            if new_path.exists():
                raw = None
            else:
                try:
                    self.check(self.cache_dir)
                    raw = self.copy(path, new_path)
                except RuntimeError as e:
                    print(e)
                    print('\n'.join([
                        "#" * 79,
                        f"WARNING: Not enough space left on {self.cache_dir}. Disable caching."
                        "#" * 79,
                    ]))
                    raw = None
                    self.cache_full = True
                    new_path = path

            self.mapping[path] = new_path
            return self.mapping[path], raw

    @contextlib.contextmanager
    def open(self, file, mode='r'):
        file, raw = self.buffer(file)
        file: Path
        if raw is None:
            with open(file, mode=mode) as fd:
                yield fd
        else:
            ext = file.suffix
            if mode in ['rb']:
                file_descriptor = io.BytesIO(raw)
            elif mode in ['r']:
                file_descriptor = io.StringIO(raw)
            else:
                raise ValueError(ext, file)
            file_descriptor.name = str(file)
            yield file_descriptor

    def __call__(self, file: Path, unsafe=False, **kwargs):
        file, raw = self.buffer(file)
        if raw is None:
            return pb.io.load(file, unsafe=unsafe, **kwargs)
        else:
            ext = file.suffix
            if ext in ['.wav']:
                file_descriptor = io.BytesIO(raw)
            elif ext in ['.json']:
                file_descriptor = io.StringIO(raw)
            else:
                # Add more "ext", when necessary
                raise NotImplementedError(ext, file)
            return pb.io.load(file_descriptor, ext=ext, unsafe=unsafe, **kwargs)

    def recursive(
            self,
            obj,
            *,
            list_to: str = 'array',
            ignore_type_error=False,
            **kwargs,
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
            self.recursive,
            ignore_type_error=ignore_type_error,
            **kwargs,
        )

        if isinstance(obj, (io.IOBase, str, Path)):
            return self(obj, **kwargs)
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

    def __del__(self):
        if self.clear and self.use_cache and self.cache_dir.exists():
            import shutil
            if os.name == 'nt':
                shutil.rmtree(self.cache_dir)
            else:
                shutil.rmtree(self.cache_dir)


@dataclasses.dataclass
class _StatisticsDiskCacheLoader(DiskCacheLoader):
    """
    Helper class for Doctest to track the function calls.
    """

    statistics: collections.Counter = dataclasses.field(
        default_factory=collections.Counter, init=False, repr=False)

    def check(self, cache_dir):
        self.statistics['check enough disk space'] += 1
        super().check(cache_dir)

    def copy(self, old_path, new_path):
        self.statistics['copy calls'] += 1
        self.statistics[f'coppied {old_path.suffix}'] += 1
        return super().copy(old_path, new_path)

    def open(self, file, mode='r'):
        self.statistics['open calls'] += 1
        self.statistics[f'opened {file.suffix}'] += 1
        return super().open(file, mode=mode)

    def buffer(self, path):
        self.statistics['buffer calls'] += 1
        return super(_StatisticsDiskCacheLoader, self).buffer(path)
