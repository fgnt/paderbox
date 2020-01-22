import io
from pathlib import Path

__all__ = [
    'dump_yaml',
    'dumps_yaml',
    'load_yaml',
    'loads_yaml',
    'load_yaml_unsafe',
    'loads_yaml_unsafe',
]


def dump_yaml(
        obj, path,
        *,
        create_path=True,
        sort_keys=False,
        **kwargs,
):
    """
    Safe dump as yaml.

    Uses `paderbox.io.dumps_json` and `paderbox.io.loads_json` to convert
    some special types (e.g. `tuple`, `pathlib.Path`, `numpy.ndarray`) to yaml
    default types (e.g. `list`, `str`, `list`). Since json is faster than yaml
    in order of magnitudes, the overhead to call json dumps and loads can be
    neglected for the runtime.

    Differences from yaml.dump:
        - pathlib.Path is converted to a str
        - sort_keys is disabled (since CPython 3.6 has the dict in order)

    Args:
        obj: Arbitrary object that is safe yaml serializable
             (see pyyaml.safe_dump).
        path: String or ``pathlib.Path`` object.
        create_path:
        **kwargs:

    >>> d = {'a': [1, 2], 'b': (3, 4), 'p': Path('abc')}
    >>> print(dumps_yaml(d))
    a:
    - 1
    - 2
    b:
    - 3
    - 4
    p: abc
    <BLANKLINE>

    """
    import yaml
    from paderbox.io import loads_json, dumps_json

    kwargs['sort_keys'] = sort_keys

    # Convert many unsafe objects to safe objects
    obj = loads_json(dumps_json(obj, sort_keys=False))

    if isinstance(path, io.IOBase):
        yaml.safe_dump(obj, path, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        if create_path:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            yaml.safe_dump(obj, f, **kwargs)
    else:
        raise TypeError(path)


def dump_yaml_unsafe(
        obj, path,
        *,
        create_path=True,
        sort_keys=False,
        **kwargs,
):
    """
    Safe dump as yaml.
    Differences from yaml.dump:
        - pathlib.Path is converted to a str
        - sort_keys is disabled (since CPython 3.6 has the dict in order)

    Args:
        obj: Arbitrary object that is safe yaml serializable
             (see pyyaml.safe_dump).
        path: String or ``pathlib.Path`` object.
        create_path:
        **kwargs:

    >>> d = {'a': [1, 2], 'b': (3, 4), 'p': Path('abc')}
    >>> print(dumps_yaml_unsafe(d))
    a:
    - 1
    - 2
    b: !!python/tuple
    - 3
    - 4
    p: !!python/object/apply:pathlib.PosixPath
    - abc
    <BLANKLINE>

    """
    import yaml
    from paderbox.io import loads_json, dumps_json

    kwargs['sort_keys'] = sort_keys

    if isinstance(path, io.IOBase):
        yaml.dump(obj, path, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        if create_path:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            yaml.dump(obj, f, **kwargs)
    else:
        raise TypeError(path)


def dumps_yaml(obj, **kwargs):
    fd = io.StringIO()
    dump_yaml(obj, path=fd, create_path=False, **kwargs)
    return fd.getvalue()


def dumps_yaml_unsafe(obj, **kwargs):
    fd = io.StringIO()
    dump_yaml_unsafe(obj, path=fd, create_path=False, **kwargs)
    return fd.getvalue()


def load_yaml(path):
    """
    Args:
        path: String or ``pathlib.Path`` object.

    >>> d = {'a': 1, 'b': 2, 'p': Path('abc')}
    >>> print(loads_yaml(dumps_yaml(d)))
    {'a': 1, 'b': 2, 'p': 'abc'}

    """
    import yaml

    if isinstance(path, io.IOBase):
        return yaml.safe_load(path)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()
        with path.open() as f:
            return yaml.safe_load(f)
    else:
        raise TypeError(path)


def load_yaml_unsafe(path):
    """
    Args:
        path: String or ``pathlib.Path`` object.

    >>> d = {'a': 1, 'b': 2, 'p': Path('abc')}
    >>> print(loads_yaml(dumps_yaml(d)))
    {'a': 1, 'b': 2, 'p': 'abc'}

    """
    import yaml

    if isinstance(path, io.IOBase):
        return yaml.unsafe_load(path)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()
        with path.open() as f:
            return yaml.unsafe_load(f)
    else:
        raise TypeError(path)


def loads_yaml(string):
    import yaml
    return yaml.safe_load(string)


def loads_yaml_unsafe(string):
    import yaml
    return yaml.unsafe_load(string)
