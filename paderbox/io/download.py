
import os
import socket
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


def download_file(remote_file, local_file, exist_ok=False):
    """
    Download single file to local_dir

    Args:
        remote_file:
        local_file:
        exist_ok:

    Returns:

    """
    local_file = Path(local_file)
    if not local_file.exists():
        def progress_hook(t):
            """
            https://raw.githubusercontent.com/tqdm/tqdm/master/examples/tqdm_wget.py
            
            Wraps tqdm instance. Don't forget to close() or __exit__()
            the tqdm instance once you're done with it (easiest using
            `with` syntax).
            """

            last_b = 0

            def inner(b=1, bsize=1, tsize=None):
                """
                b  : int, optional
                    Number of blocks just transferred [default: 1].
                bsize  : int, optional
                    Size of each block (in tqdm units) [default: 1].
                tsize  : int, optional
                    Total size (in tqdm units). If [default: None]
                    remains unchanged.
                """
                nonlocal last_b
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b) * bsize)
                last_b = b

            return inner

        tmp_file = str(local_file) + '.tmp'
        with tqdm(
                desc="{0: >25s}".format(Path(remote_file).stem),
                file=sys.stdout,
                unit='B',
                unit_scale=True,
                miniters=1,
                leave=False,
                ascii=True
        ) as t:
            urlretrieve(
                str(remote_file),
                filename=tmp_file,
                reporthook=progress_hook(t),
                data=None
            )
        os.rename(tmp_file, local_file)
    elif not exist_ok:
        raise FileExistsError(local_file)
    return local_file


def extract_file(local_file, exist_ok=False):
    """
    If local_file is .zip or .tar.gz files are extracted.

    Args:
        local_file:
        exist_ok:

    Returns:

    """
    local_file = Path(local_file)
    local_dir = local_file.parent
    if local_file.exists():

        if local_file.name.endswith('.zip'):
            with zipfile.ZipFile(local_file, "r") as z:
                # Start extraction
                members = z.infolist()
                for i, member in enumerate(members):
                    target_file = local_dir / member.filename
                    if not target_file.exists():
                        try:
                            z.extract(member=member, path=local_dir)
                        except KeyboardInterrupt:
                            # Delete latest file, since most likely it
                            # was not extracted fully
                            if target_file.exists():
                                os.remove(target_file)
                            raise
                    elif not exist_ok:
                        raise FileExistsError(target_file)
            os.remove(local_file)

        elif local_file.name.endswith('.tar.gz'):
            with tarfile.open(local_file, "r:gz") as tar:
                for i, tar_info in enumerate(tar):
                    target_file = local_dir / tar_info.name
                    if not target_file.exists():
                        try:
                            tar.extract(tar_info, local_dir)
                        except KeyboardInterrupt:
                            # Delete latest file, since most likely it
                            # was not extracted fully
                            if target_file.exists():
                                os.remove(target_file)
                            raise
                    elif not exist_ok:
                        raise FileExistsError(target_file)
                    tar.members = []
            os.remove(local_file)


def download_file_list(file_list, target_dir, exist_ok=False, logger=None):
    """
    Download file_list to target_dir

    Args:
        file_list:
        target_dir:
        exist_ok:
        logger:

    Returns:

    """

    target_dir = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    item_progress = tqdm(
        file_list, desc="{0: <25s}".format('Download files'),
        file=sys.stdout, leave=False, ascii=True)

    local_files = list()
    for remote_file in item_progress:
        local_files.append(
            download_file(
                remote_file,
                target_dir / Path(remote_file).name,
                exist_ok=exist_ok
            )
        )

    item_progress = tqdm(
        local_files,
        desc="{0: <25s}".format('Extract files'),
        file=sys.stdout,
        leave=False,
        ascii=True
    )

    if logger is not None:
        logger.info('Starting Extraction')
    for _id, local_file in enumerate(item_progress):
        if local_file and local_file.exists():
            if logger is not None:
                logger.info(
                    '  {title:<15s} [{item_id:d}/{total:d}] {package:<30s}'
                    .format(
                        title='Extract files ',
                        item_id=_id,
                        total=len(item_progress),
                        package=local_file
                    )
                )
            extract_file(local_file, exist_ok=exist_ok)
