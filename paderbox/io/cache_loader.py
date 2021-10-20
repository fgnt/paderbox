import dataclasses
from pathlib import Path
import collections
import io
import os.path
import contextlib

import paderbox as pb


def check(cache_dir):
    # copied from lazy_dataset.core.DiskCacheDataset
    import shutil
    diskusage = shutil.disk_usage(cache_dir)
    if diskusage.free < 5 * 1024 ** 3:
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
        Be carefull, that it is not guaranteed, that the files are afterwards
        deleted, because some errors don't trigger the shutdown of the python
        process and simply exit the program. e.g. pressing Ctrl-C multiple
        times may leafe some artefacts on the filesystem.


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
    mapping: dict = dataclasses.field(default_factory=dict, init=False)
    clear: bool = True

    def __post_init__(self):
        if len(os.listdir(self.cache_dir)) != 0:
            raise RuntimeError(f'Dir {self.cache_dir} is not empty.')

    def check(self, cache_dir):
        return check(cache_dir)

    def copy(self, old_path, new_path, keep_bytes=True):
        if not keep_bytes:
            import shutil
            shutil.copy(old_path, new_path)
        else:
            with old_path.open(mode='rb') as f:
                raw = f.read()
            try:
                with new_path.open(mode='wb') as f:
                    f.write(raw)
            except FileNotFoundError:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                with new_path.open(mode='wb') as f:
                    f.write(raw)
            return raw

    def buffer(self, path, keep_bytes=True):
        path = Path(path).absolute()
        if path in self.mapping:
            return self.mapping[path], None
        else:
            self.check(self.cache_dir)
            new_path = Path(f'{self.cache_dir}/{path}')
            raw = self.copy(path, new_path, keep_bytes)
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

    def __call__(self, file, unsafe=False):
        file, raw = self.buffer(file, keep_bytes=True)
        if raw is None:
            return pb.io.load(file, unsafe=unsafe)
        else:
            ext = file.suffix
            if ext in ['.wav']:
                file_descriptor = io.BytesIO(raw)
            elif ext in ['.json']:
                file_descriptor = io.StringIO(raw)
            else:
                raise ValueError(ext, file)
            return pb.io.load(file_descriptor, ext=ext, unsafe=unsafe)

    def __del__(self):
        if self.clear:
            import shutil
            shutil.rmtree(self.cache_dir)


@dataclasses.dataclass
class _StatisticsDiskCacheLoader(DiskCacheLoader):
    """

    """

    statistics: collections.Counter = dataclasses.field(default_factory=collections.Counter, init=False, repr=False)

    def check(self, cache_dir):
        self.statistics['check enough disk space'] += 1
        super().check(cache_dir)

    def copy(self, old_path, new_path, keep_bytes=True):
        self.statistics['copy calls'] += 1
        self.statistics[f'coppied {old_path.suffix}'] += 1
        return super().copy(old_path, new_path, keep_bytes)

    def open(self, file, mode='r'):
        self.statistics['open calls'] += 1
        self.statistics[f'opened {file.suffix}'] += 1
        return super().open(file, mode=mode)

    def buffer(self, path, keep_bytes=True):
        self.statistics['buffer calls'] += 1
        assert keep_bytes is True, keep_bytes
        return super(_StatisticsDiskCacheLoader, self).buffer(path, keep_bytes)
