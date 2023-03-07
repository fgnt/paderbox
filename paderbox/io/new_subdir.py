import os
import re
import time
import fnmatch
import datetime
import urllib.request
from pathlib import Path


def _removeprefix(self: str, prefix: str) -> str:
    # Backport from Python 3.9: https://www.python.org/dev/peps/pep-0616/
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]


def _removesuffix(self: str, suffix: str) -> str:
    # Backport from Python 3.9:  https://www.python.org/dev/peps/pep-0616/
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]


def get_new_subdir(
        basedir: [str, Path],
        *,
        id_naming: [str, callable]='index',
        mkdir: bool=True,
        prefix: str=None,
        suffix: str=None,
        consider_mpi: bool=False,
        dry_run: bool=False,
):
    """Determine a new non-existent sub directory.

    Features:
     - With mkdir: Thread and process save.
     - Different conventions for ID naming possible, default running index.
     - MPI aware: Get the folder on one worker and distribute to others.

    Args:
        basedir:
            The new subdir will be inside this directory
        id_naming:
            The id naming that is used for the folder name.
             - str: 'index':
                The largest index in basedir + 1.
                e.g.: '1', '2', ...
             - str: 'time': A timestamp with the format %Y-%m-%d-%H-%M-%S
                e.g. '2020-08-13-17-02-57'
             - callable: Each call should generate a new name.
        mkdir:
            Creates the dir and makes the program process/thread safe.
            Note this option ensures that you don't get a
            conflict between two concurrent calls of get_new_subdir.
            Example:
                You launch several times your programs and each should get
                another folder (e.g. hyperparameter search). When inspecting
                basedir maybe some recognize they can use '2' as sub folder.
                This option ensures, that only one program gets the '2' and the
                remaining programs search for another free id.
        prefix:
            Optional prefix for the id. e.g.: '2' -> '{prefix}_2'
        suffix:
            Optional suffix for the id. e.g.: '2' -> '2_{suffix}'
        consider_mpi:
            If True, only search on one mpi process for the folder and
            distribute the folder name.
            When using mpi (and `consider_mpi is False`) the following can/will
            happen
             - When mkdir is True every process will get another folder.
               i.e. each process has a folder just for this process.
             - Warning: Never use mpi, when `mkdir is False` and
               `consider_mpi is False`. Depending on some random factors
               (e.g. python startup time) all workers could get the same
               folder, but mostly some get the same folder and some different.
               You never want this.
        dry_run:
            When true, disables mkdir and prints the folder name.

    Returns:
        pathlib.Path of the new subdir

    >>> # root folder usually contain no digits
    >>> get_new_subdir('/', dry_run=True)    # doctest: +ELLIPSIS
    dry_run: "os.mkdir(...1)"
    ...Path('...1')

    >>> import numpy as np
    >>> np.random.seed(0)  # This is for doctest. Never use it in practise.
    >>> get_new_subdir('/', id_naming=NameGenerator(), dry_run=True)  # doctest: +ELLIPSIS
    dry_run: "os.mkdir(...nice_tomato_fox)"
    ...Path('...nice_tomato_fox')
    """

    if consider_mpi:
        import dlp_mpi
        if dlp_mpi.IS_MASTER:
            pass
        else:
            new_folder = None
            new_folder = dlp_mpi.bcast(new_folder)
            return new_folder

    basedir = Path(basedir).expanduser().resolve()
    if not basedir.exists():
        if dry_run:
            print(f'dry_run: "os.makedirs({basedir})"')
            # ToDo: Make this working.
            #       Will fail when calling os.listdir
        else:
            basedir.mkdir(parents=True)

    if Path('/net') in basedir.parents:
        # If nt filesystem, assert not in /net/home
        assert Path('/net/home') not in basedir.parents, basedir

    prefix_ = f'{prefix}_' if prefix else ''
    _suffix = f'_{suffix}' if suffix else ''

    for i in range(200):
        if id_naming == 'index':
            if prefix is None and suffix is None:
                dir_nrs = [
                    int(d)
                    for d in os.listdir(str(basedir))
                    if (basedir / d).is_dir() and d.isdigit()
                ]
                _id = max(dir_nrs + [0]) + 1
            else:
                def remove_pre_suf(d):
                    return _removesuffix(
                        _removeprefix(str(d), prefix_),
                        _suffix
                    )

                dir_nrs = [
                    int(remove_pre_suf(d))
                    for d in os.listdir(str(basedir))
                    if (basedir / d).is_dir()
                    if fnmatch.fnmatch(d, f'{prefix_}*{_suffix}')
                    if remove_pre_suf(d).isdigit()
                ]
                dir_nrs += [0]
                _id = max(dir_nrs) + 1
                _id = f'{prefix_}{_id}{_suffix}'
        elif id_naming == 'time':
            if i != 0:
                time.sleep(1)
            _id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            _id = f'{prefix_}{_id}{_suffix}'
        elif callable(id_naming):
            _id = id_naming()
        else:
            raise ValueError(id_naming)

        simu_dir = basedir / str(_id)

        try:
            if dry_run:
                print(f'dry_run: "os.mkdir({simu_dir})"')
            elif mkdir is False:
                pass
            elif mkdir is True:
                simu_dir.mkdir()
            else:
                raise ValueError(mkdir)

            if consider_mpi:
                import dlp_mpi
                assert dlp_mpi.IS_MASTER, dlp_mpi.RANK
                simu_dir = dlp_mpi.bcast(simu_dir)

            return simu_dir
        except FileExistsError:
            # Catch race conditions
            if i > 100:
                # After some tries,
                # expect that something other went wrong
                raise


def _get_list_from_unique_names_generator(name_type, overwrite_cache=False):
    """
    Download a list from
    https://github.com/andreasonny83/unique-names-generator
    and cache the result in `~/.cache/padertorch/unique_names_generator`.

    >>> _get_list_from_unique_names_generator('adjectives')[:3]
    ['able', 'above', 'absent']
    >>> _get_list_from_unique_names_generator('colors')[:3]
    ['amaranth', 'amber', 'amethyst']
    >>> _get_list_from_unique_names_generator('animals')[:3]
    ['aardvark', 'aardwolf', 'albatross']
    >>> _get_list_from_unique_names_generator('names')[:3]
    ['Aaren', 'Aarika', 'Abagael']
    >>> _get_list_from_unique_names_generator('countries')[:3]
    ['Afghanistan', 'Åland Islands', 'Albania']
    >>> _get_list_from_unique_names_generator('star-wars')[:3]
    ['Ackbar', 'Adi Gallia', 'Anakin Skywalker']

    """
    # adjectives.ts
    # animals.ts
    # colors.ts
    # countries.ts
    # names.ts
    # star-wars.ts
    # index.ts    -> Does not work (Is not dictionary)
    # numbers.ts  -> Does not work (Cannot be parsed)
    from platformdirs import user_cache_dir
    import paderbox as pb

    file = (
            Path(user_cache_dir('paderbox'))
            / 'unique_names_generator' / f'{name_type}.json'
    )
    if overwrite_cache or (not file.exists()):
        url = (
            f'https://raw.githubusercontent.com/andreasonny83/'
            f'unique-names-generator/v4.6.0/src/dictionaries/{name_type}.ts'
        )

        try:
            resource = urllib.request.urlopen(url)
        except Exception as e:
            # ToDo: use the following api to list the names
            # https://api.github.com/repos/andreasonny83/unique-names-generator/git/trees/main?recursive=1
            raise ValueError(
                f'Tried to download {name_type!r}.\nCould not open\n{url}\n'
                'See in\n'
                'https://github.com/andreasonny83/unique-names-generator/tree/v4.6.0/src/dictionaries\n'
                'for valid names.'
            )
        # https://stackoverflow.com/a/19156107/5766934
        data = resource.read().decode(resource.headers.get_content_charset())

        data = re.findall("'(.*?)'", data)

        pb.io.dump(data, file)
    else:
        data = pb.io.load(file)
    return data


class NameGenerator:
    """
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> ng = NameGenerator()
    >>> ng()
    'nice_tomato_fox'
    >>> ng.possibilities()  # With 22 million a collision is unlikely
    22188920
    >>> ng = NameGenerator(['adjectives', 'animals'])
    >>> ng()
    'regional_prawn'

    """
    def __init__(
            self,
            lists=('adjectives', 'colors', 'animals'),
            separator='_',
            rng=None,
            replace=True,
            # style='default',  # 'capital', 'upperCase', 'lowerCase'
    ):
        self.lists = [
            _get_list_from_unique_names_generator(l)
            if isinstance(l, str) else list(map(str, l))
            for l in lists
        ]
        self.separator = separator
        if rng is None:
            import numpy as np
            rng = np.random
        self.rng = rng
        self.replace = replace
        self.seen = set()

    def __call__(self):
        for _ in range(1000):
            name_parts = [
                self.rng.choice(d)
                for d in self.lists
            ]
            name = self.separator.join(name_parts)

            if self.replace:
                break
            elif name in self.seen:
                continue
            else:
                self.seen.add(name)
                break
        else:
            raise RuntimeError("Couldn't find a new name.")

        return name

    def possibilities(self):
        import numpy as np
        return np.prod([[len(d) for d in self.lists]])


if __name__ == '__main__':
    def cli(
            basedir,
            *,
            id_naming='index',
            mkdir=True,
            prefix=None,
            suffix=None,
            consider_mpi=False,  # useless for cli?
            dry_run=False,
    ):
        """

        Args:
            basedir:
            id_naming: e.g. 'index', 'time', 'adjective_color_animal'
            mkdir:
            prefix:
            suffix:
            consider_mpi:
            dry_run:

        Returns:

        """
        # Add more cli mappings if necessary
        if id_naming == 'adjective_color_animal':
            id_naming = NameGenerator(('adjectives', 'colors', 'animals'))
        return str(get_new_subdir(**locals()))

    import fire
    fire.Fire(cli)
