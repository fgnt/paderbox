
import os
import socket
import sys
import tarfile
import zipfile
from os import path
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
    if not path.isfile(local_file):
        def progress_hook(t):
            """
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

        tmp_file = local_file + '_tmp_file'
        with tqdm(
                desc="{0: >25s}".format(
                    path.splitext(remote_file.split('/')[-1])[0]
                ),
                file=sys.stdout,
                unit='B',
                unit_scale=True,
                miniters=1,
                leave=False,
                ascii=True
        ) as t:
            urlretrieve(
                remote_file,
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
    If local_file is .zip a common first level folder is omitted.

    Args:
        local_file:
        exist_ok:

    Returns:

    """
    local_dir = path.dirname(local_file)
    if path.isfile(local_file):

        if local_file.endswith('.zip'):

            with zipfile.ZipFile(local_file, "r") as z:
                # Trick to omit first level folder
                parts = []
                for name in z.namelist():
                    if not name.endswith('/'):
                        parts.append(name.split('/')[:-1])
                prefix = path.commonprefix(parts)
                prefix = prefix[0] + '/' if prefix else ''
                offset = len(prefix)

                # Start extraction
                members = z.infolist()
                for i, member in enumerate(members):
                    if member.filename != prefix:
                        member.filename = member.filename[offset:]
                        if not path.isfile(
                                path.join(local_dir, member.filename)
                        ):
                            try:
                                z.extract(member=member, path=local_dir)
                            except KeyboardInterrupt:
                                # Delete latest file, since most likely it
                                # was not extracted fully
                                os.remove(
                                    path.join(local_dir, member.filename)
                                )

                                # Quit
                                sys.exit()
                        elif not exist_ok:
                            raise FileExistsError(path.join(local_dir, member.filename))
            os.remove(local_file)

        elif local_file.endswith('.tar.gz'):
            with tarfile.open(local_file, "r:gz") as tar:
                for i, tar_info in enumerate(tar):
                    if not path.isfile(
                        path.join(local_dir, tar_info.name)
                    ):
                        tar.extract(tar_info, local_dir)
                    elif not exist_ok:
                        raise FileExistsError(path.join(local_dir, tar_info.name))
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

    os.makedirs(target_dir, exist_ok=True)

    # Set socket timeout
    socket.setdefaulttimeout(120)

    item_progress = tqdm(
        file_list, desc="{0: <25s}".format('Download files'),
        file=sys.stdout, leave=False, ascii=True)

    local_files = list()
    for remote_file in item_progress:
        local_files.append(
            download_file(
                remote_file,
                path.join(target_dir, path.basename(remote_file)),
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
        if local_file and path.isfile(local_file):
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
