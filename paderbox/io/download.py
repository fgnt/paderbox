
import os
import tarfile
import zipfile
import warnings
from pathlib import Path
from urllib.request import urlretrieve
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def download_file(remote_file, local_file, exist_ok=False, extract=False):
    """
    Download single file to local_dir

    Args:
        remote_file:
        local_file:
        exist_ok:
        extract:
        progress_par:

    Returns:

    """
    local_file = Path(local_file)
    if not local_file.exists():
        tmp_file = str(local_file) + '.tmp'
        urlretrieve(
            str(remote_file),
            filename=tmp_file,
            data=None
        )
        os.rename(tmp_file, local_file)
    elif not exist_ok:
        raise FileExistsError(local_file)
    if extract:
        extract_file(local_file, exist_ok=exist_ok)
    return local_file


def extract_file(local_file, target_dir=None, exist_ok=False):
    """
    If local_file is .zip or .tar.gz files are extracted.

    Args:
        local_file:
        target_dir:
        exist_ok:

    Returns:

    """
    local_file = Path(local_file)
    assert local_file.exists(), local_file
    if target_dir is None:
        target_dir = local_file.parent
    else:
        target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if local_file.name.endswith('.zip'):
        with zipfile.ZipFile(local_file, "r") as z:
            # Start extraction
            members = z.infolist()
            for i, member in enumerate(members):
                target_file = target_dir / member.filename
                if not target_file.exists():
                    try:
                        z.extract(member=member, path=target_dir)
                    except KeyboardInterrupt:
                        # Delete latest file, since most likely it
                        # was not extracted fully
                        if target_file.exists():
                            os.remove(target_file)
                        raise
                elif not exist_ok:
                    raise FileExistsError(target_file)
        os.remove(local_file)

    elif local_file.name.endswith('.tar.gz') or local_file.name.endswith('.tar'):
        mode = "r:gz" if local_file.name.endswith('.tar.gz') else "r"
        with tarfile.open(local_file, mode) as tar:
            for i, tar_info in enumerate(tar):
                target_file = target_dir / tar_info.name
                if not target_file.exists():
                    try:
                        tar.extract(tar_info, target_dir)
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
    else:
        warnings.warn("Unsupported file format: Cannot extract file.")


def download_file_list(file_list, target_dir, extract=True, exist_ok=False, num_workers=1):
    """
    Download file_list to target_dir

    Args:
        file_list:
        target_dir:
        exist_ok:
        extract:
        num_workers:

    Returns:

    """
    target_dir = Path(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    pbar = tqdm(initial=0, total=len(file_list))

    if isinstance(extract, bool):
        extract = len(file_list) * [extract]
    assert len(extract) == len(file_list), (len(extract), len(file_list))
    if num_workers > 1:
        with ProcessPoolExecutor(num_workers) as ex:
            for _ in ex.map(
                download_file,
                file_list,
                [target_dir / Path(f).name.split('?')[0] for f in file_list], # extract file names from urls discarding query strings
                len(file_list) * [exist_ok],
                extract,
            ):
                pbar.update(1)
    else:
        for _ in map(
            download_file,
            file_list,
            [target_dir / Path(f).name.split('?')[0] for f in file_list],
            len(file_list) * [exist_ok],
            extract,
        ):
            pbar.update(1)
