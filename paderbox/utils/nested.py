import copy

import collections
import itertools
import operator
from typing import Optional, Any, Union, Sequence, Mapping, Tuple, Callable, Generator, Iterable, Iterator


def flatten(d, sep: Optional[str] = '.', *, flat_type=dict):
    """
    Flatten a nested `dict` using a specific separator.

    Args:
        sep: When `None`, return `dict` with `tuple` keys (guarantees inversion
                of flatten) else join the keys with sep
        flat_type:  Allow other mappings instead of `flat_type` to be
                flattened, e.g. using an isinstance check.

    import collections
    flat_type=collections.abc.MutableMapping

    >>> d_in = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}
    >>> d = flatten(d_in)
    >>> for k, v in d.items(): print(k, v)
    a 1
    c.a 2
    c.b.x 5
    c.b.y 10
    d [1, 2, 3]
    >>> d = flatten(d_in, sep='_')
    >>> for k, v in d.items(): print(k, v)
    a 1
    c_a 2
    c_b_x 5
    c_b_y 10
    d [1, 2, 3]
    """

    # https://stackoverflow.com/a/6027615/5766934

    # {k: v for k, v in d.items()}

    def inner(d, parent_key):
        items = {}
        for k, v in d.items():
            new_key = parent_key + (k,)
            if isinstance(v, flat_type) and v:
                items.update(inner(v, new_key))
            else:
                items[new_key] = v
        return items

    items = inner(d, ())
    if sep is None:
        return items
    else:
        return {
            sep.join(k): v for k, v in items.items()
        }


def deflatten(d: dict, sep: Optional[str] = '.', maxdepth: int = -1):
    """
    Build a nested `dict` from a flat dict respecting a separator.

    Args:
        d: Flattened `dict` to reconstruct a `nested` dict from
        sep: The separator used in the keys of `d`. If `None`, `d.keys()` should
            only contain `tuple`s.
        maxdepth: Maximum depth until wich nested conversion is performed

    >>> d_in = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]}
    >>> d = flatten(d_in)
    >>> for k, v in d.items(): print(k, v)
    a 1
    c.a 2
    c.b.x 5
    c.b.y 10
    d [1, 2, 3]
    >>> deflatten(d)
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y': 10}}, 'd': [1, 2, 3]}
    >>> deflatten(d, maxdepth=1)
    {'a': 1, 'c': {'a': 2, 'b.x': 5, 'b.y': 10}, 'd': [1, 2, 3]}
    >>> deflatten(d, maxdepth=0)
    {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    >>> d = flatten(d_in, sep='_')
    >>> for k, v in d.items(): print(k, v)
    a 1
    c_a 2
    c_b_x 5
    c_b_y 10
    d [1, 2, 3]
    >>> deflatten(d, sep='_')
    {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y': 10}}, 'd': [1, 2, 3]}
    >>> deflatten({('a', 'b'): 'd', ('a', 'c'): 'e'}, sep=None)
    {'a': {'b': 'd', 'c': 'e'}}
    >>> deflatten({'a.b': 1, 'a': 2})
    Traceback (most recent call last):
      ...
    AssertionError: Conflicting keys! ('a',)
    >>> deflatten({'a': 1, 'a.b': 2})
    Traceback (most recent call last):
      ...
    AssertionError: Conflicting keys! ('a', 'b')

    """
    ret = {}
    if sep is not None:
        d = {
            tuple(k.split(sep, maxdepth)): v for k, v in d.items()
        }

    for keys, v in d.items():
        sub_dict = ret
        for sub_key in keys[:-1]:
            if sub_key not in sub_dict:
                sub_dict[sub_key] = {}
            assert isinstance(sub_dict[sub_key], dict), (
                f'Conflicting keys! {keys}'
            )
            sub_dict = sub_dict[sub_key]
        assert keys[-1] not in sub_dict, f'Conflicting keys! {keys}'
        sub_dict[keys[-1]] = v
    return ret


def nested_update(orig, update):
    # Todo:
    assert isinstance(orig, type(update)), (type(orig), type(update))
    if isinstance(orig, list):
        for i, value in enumerate(update):
            if isinstance(value, (dict, list)) \
                    and i < len(orig) and isinstance(orig[i], type(value)):
                nested_update(orig[i], value)
            elif i < len(orig):
                orig[i] = value
            else:
                assert i == len(orig), (i, len(orig))
                orig.append(value)
    elif isinstance(orig, dict):
        for key, value in update.items():
            if isinstance(value, (dict, list)) \
                    and key in orig and isinstance(orig[key], type(value)):
                nested_update(orig[key], value)
            else:
                orig[key] = value


def nested_merge(default_dict, *update_dicts, allow_update=True, inplace=False):
    """
    Nested updates the first dict with all other dicts.

    The last dict has the highest priority when `allow_update` is `True`.
    When `allow_update` is `False`, it is assumed that no values get
    overwritten and an exception is raised when duplicate keys are found.

    When `inplace` is `True`, the `default_dict` is manipulated inplace. This is
    useful for `collections.defaultdict`.

    # Example from: https://stackoverflow.com/q/3232943/5766934
    >>> dictionary = {'level1': {'level2': {'levelA': 0,'levelB': 1}}}
    >>> update = {'level1': {'level2': {'levelB': 10, 'levelC': 2}}}
    >>> new = nested_merge(dictionary, update)
    >>> print(new)
    {'level1': {'level2': {'levelA': 0, 'levelB': 10, 'levelC': 2}}}
    >>> print(dictionary)  # no inplace manipulation
    {'level1': {'level2': {'levelA': 0, 'levelB': 1}}}
    >>> new = nested_merge(dictionary, update, inplace=True)
    >>> print(dictionary)  # with inplace manipulation
    {'level1': {'level2': {'levelA': 0, 'levelB': 10, 'levelC': 2}}}

    >>> nested_merge({'foo': 0}, {'foo': {'bar':1}})
    {'foo': {'bar': 1}}

    >>> nested_merge({'foo': {'bar': 1}}, {'foo': 0})
    {'foo': 0}

    >>> nested_merge({'foo': {'bar': 1}}, {'foo': 0}, allow_update=False)
    Traceback (most recent call last):
    ...
    AssertionError: [{'bar': 1}, 0]
    >>> nested_merge({'foo': {'bar': 1}}, {'blub': 0}, allow_update=False)
    {'foo': {'bar': 1}, 'blub': 0}
    >>> nested_merge({'foo': {'blub': 0}}, {'foo': 1}, {'foo': {'bar': 1}})
    {'foo': {'bar': 1}}
    >>> nested_merge({'foo': 1}, {'foo': {'bar': 1}}, allow_update=False)
    Traceback (most recent call last):
    ...
    AssertionError: [1, {'bar': 1}]
    >>> nested_merge({'foo': {'bar': 1}}, {'foo': {'bar': 1}}, allow_update=False)
    {'foo': {'bar': 1}}
    >>> nested_merge({'foo': {'bar': 1}}, {'foo': {'blub': 1}}, allow_update=False)
    {'foo': {'bar': 1, 'blub': 1}}


    """
    if len(update_dicts) == 0:
        if inplace:
            return default_dict
        else:
            return copy.copy(default_dict)

    dicts = [default_dict, *update_dicts]

    def get_value_for_key(key):
        values = [
            d[key]
            for d in dicts
            if key in d.keys()
        ]
        if isinstance(values[-1], collections.abc.Mapping):
            mapping_values = []
            for value in values[::-1]:
                if isinstance(value, collections.abc.Mapping):
                    mapping_values.insert(0, value)
                else:
                    break
            if not allow_update:
                assert len(mapping_values) == len(values), values
            return nested_merge(
                *mapping_values, allow_update=allow_update, inplace=inplace)
        else:
            if not allow_update:
                try:
                    values = set(values)
                except TypeError:
                    # set requires hashable, force len 1 when not hashable
                    # e.g.: TypeError: unhashable type: 'dict'
                    pass
                assert len(values) == 1, values
                values = list(values)
            return values[-1]

    keys = itertools.chain(*[
        d.keys()
        for d in dicts
    ])

    if not inplace:
        default_dict = copy.copy(default_dict)

    for k in keys:
        default_dict[k] = get_value_for_key(k)

    return default_dict


def nested_op(
        func,
        arg1, *args,
        broadcast=False,
        handle_dataclass=False,
        keep_type=True,
        mapping_type=collections.abc.Mapping,
        sequence_type=(tuple, list),
):
    """
    Applies the function `func` to the leafs of the nested data structures in
    `arg1` and `*args`.
    This is similar to the map function that applies the function the the
    elements of an iterable input (e.g. list).
    This function is `nested_map` with a fancy name.

    CB: Should handle_dataclass be True or False?
        Other suggestions for the name "handle_dataclass"?

    >>> import operator
    >>> nested_op(operator.add, (3, 5), (7, 11))  # like map
    (10, 16)
    >>> nested_op(operator.add, {'a': (3, 5)}, {'a': (7, 11)})  # with nested
    {'a': (10, 16)}

    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=[1], b=dict(c=4)), dict(a=[0], b=dict(c=1)))
    {'a': [1], 'b': {'c': 7}}
    >>> arg1, arg2 = dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3])
    >>> nested_op(\
    lambda x, y: x + 3*y, arg1, arg2)
    Traceback (most recent call last):
    ...
    AssertionError: ({'c': [1, 1]}, ([1, 3],))

    Note the broadcasting behavior (arg2.b is broadcasted to arg2.b.c)
    >>> nested_op(\
    lambda x, y: x + 3*y, arg1, arg2, broadcast=True)
    {'a': 1, 'b': {'c': [4, 10]}}

    >>> import dataclasses
    >>> @dataclasses.dataclass
    ... class Data:
    ...     a: int
    ...     b: int
    >>> nested_op(operator.add, Data(3, 5), Data(7, 11), handle_dataclass=True)
    Data(a=10, b=16)

    Args:
        func:
        arg1:
        *args:
        broadcast:
        handle_dataclass: Treat dataclasses as "nested" type or not
        keep_type: Keep the types in the nested structure of arg1 for the
            output or use dict and list as types for the output.
        mapping_type: Types that are interpreted as mapping.
        sequence_type: Types that are interpreted as sequence.

    Returns:

    """
    # These kwargs are forwarded to subsequent calls of nested_op
    kwargs = dict(
        broadcast=broadcast,
        mapping_type=mapping_type,
        sequence_type=sequence_type,
        keep_type=keep_type,
        handle_dataclass=handle_dataclass,
    )

    if isinstance(arg1, mapping_type):
        if not broadcast:
            assert all(
                [isinstance(arg, mapping_type) and arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        else:
            assert all(
                [not isinstance(arg, mapping_type) or arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        keys = arg1.keys()
        output = {
            key: nested_op(
                func,
                arg1[key],
                *[arg[key] if isinstance(arg, mapping_type) else arg
                  for arg in args],
                **kwargs
            )
            for key in keys
        }
        if keep_type:
            output = arg1.__class__(output)
        return output
    elif isinstance(arg1, sequence_type):
        if not broadcast:
            assert all([
                isinstance(arg, sequence_type) and len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not isinstance(arg, sequence_type) or len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        output = [
            nested_op(
                func,
                arg1[j],
                *[
                    arg[j] if isinstance(arg, sequence_type) else arg
                    for arg in args
                ],
                **kwargs
            )
            for j in range(len(arg1))
        ]
        if keep_type:
            output = arg1.__class__(output)
        return output
    elif handle_dataclass and hasattr(arg1, '__dataclass_fields__'):
        if not broadcast:
            assert all([
                hasattr(arg, '__dataclass_fields__')
                and arg.__dataclass_fields__ == arg1.__dataclass_fields__
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not hasattr(arg, '__dataclass_fields__')
                or arg.__dataclass_fields__ == arg1.__dataclass_fields__
                for arg in args
            ]), (arg1, args)
        return arg1.__class__(
            **{
                f_key: nested_op(
                    func,
                    getattr(arg1, f_key),
                    *[getattr(arg, f_key)
                      if hasattr(arg, '__dataclass_fields__')
                      else arg
                      for arg in args
                      ],
                    **kwargs
                )
                for f_key in arg1.__dataclass_fields__
            }
        )

    return func(arg1, *args)


def squeeze_nested(orig):
    """
    recursively flattens hierarchy if all nested elements have the same value

    >>> squeeze_nested({'a': 1, 'b': 1})
    1
    """
    if isinstance(orig, (dict, list)):
        keys = list(orig.keys() if isinstance(orig, dict) else range(len(orig)))
        squeezed = True
        for key in keys:
            orig[key] = squeeze_nested(orig[key])
            if isinstance(orig[key], (list, dict)):
                squeezed = False
        if squeezed and all([orig[key] == orig[keys[0]] for key in keys]):
            return orig[keys[0]]
    return orig


_NO_VALUE = object()


def get_by_path(
        container: Union[Mapping, Sequence],
        path: Union[str, Tuple[Any, ...], None],
        *,
        allow_early_stopping: bool = False,
        sep: str = '.',
        default: Any = _NO_VALUE,
) -> Any:
    """
    Gets a value from a nested `container` by the dotted path `path`.

    If `container` is a nested dictionary (and doesn't contain any lists), this
    operation with default arguments is equivalent to `flatten(container)[path]`
    for string paths, but faster.

    Args:
        container: The container to get the value from
        path: Dotted path or tuple of keys to index the nested container with.
            A `tuple` is useful if not all keys are strings. If it is a `str`,
            keys are delimited by `delimiter`
        allow_early_stopping: If `True`, "broadcast" leaves if a sub-path of
            `path` points to a leaf in `container`. Useful for nested structures
            where the exact structure can vary, e.g., in a database the number
            of samples for the "observation" can be located in "num_samples" or
            "num_samples.observation". Use with care!
        sep: The delimiter for keys in path
        default: Default value that is returned when the path is not present in
            the nested container (and cannot be broadcasted if `broadcast=True`)

    Returns:
        Value located at `path` in `container`

    Examples:
        >>> d = {'a': 'b', 'c': {'d': {'e': 'f'}, 'g': [1, [2, 3], 4]}}
        >>> get_by_path(d, 'a')
        'b'
        >>> get_by_path(d, 'c.d.e')
        'f'
        >>> get_by_path(d, ('c', 'g', 1, 0))
        2
        >>> get_by_path(d, 'a.b.c', allow_early_stopping=True)
        'b'
        >>> get_by_path(d, 'c.b.c', default=42)
        42
    """
    if path is None:
        return container

    if isinstance(path, str):
        path = path.split(sep)

    for k in path:
        try:
            container = container[k]
        except Exception as e:
            # Indexing a custom type can raise any exception, in which case
            # we try to broadcast
            # Not sure if broadcasting makes sense for lists/tuples. It is
            # hard to check for custom sequences because of str, so
            # sequences are broadcasted here
            if allow_early_stopping and not isinstance(container, Mapping):
                return container

            # We can't add another more specific except block because we have to
            # catch all exceptions for the broadcasting. Assuming here that
            # custom containers raise KeyErrors and IndexErrors correctly when
            # the indexed element is not found
            if isinstance(e, (KeyError, IndexError)):
                if default is not _NO_VALUE:
                    return default

            raise
    return container


def set_by_path(
        container: Union[Mapping, Sequence],
        path: Union[str, Tuple[Any, ...], None],
        value: Any,
        *,
        sep: str = '.',
) -> None:
    """
    Sets a value in the nested dictionary `d` by the dotted path.

    Modifies `d` inplace.

    Args:
        container: The container to get the value from
        path: Dotted path or tuple of keys to index the nested container with.
            A `tuple` is useful if not all keys are strings. If it is a `str`,
            keys are delimited by `delimiter`
        value: The value to set in `d` for `path`
        sep: The delimiter for keys in path

    Examples:
        >>> d = {}
        >>> set_by_path(d, 'a', {})
        >>> d
        {'a': {}}
        >>> set_by_path(d, 'a.b', {'c': [1, 2, 3], 'd': 'e'})
        >>> d
        {'a': {'b': {'c': [1, 2, 3], 'd': 'e'}}}
        >>> set_by_path(d, ('a', 'b', 'c', 2), 42)
        >>> d
        {'a': {'b': {'c': [1, 2, 42], 'd': 'e'}}}
    """
    if isinstance(path, str):
        path = path.split(sep)
    container = get_by_path(container, path[:-1])
    container[path[-1]] = value


def nested_iter_items(
        container: Iterable,
        *,
        iter_types: Tuple[type] = (dict, tuple, list)
) -> Generator[Tuple[Tuple[Any, ...], Any], None, None]:
    """
    Iterates over the leaves of `container`. Yield tuples of `(path, value)`,
    where path is the path to the leaf as a `tuple` of keys.

    We could consider a `nested_iter_values` that only iterates over the values
    and does not yield the paths when someone needs it and this function is too
    slow for some usecase.

    If someone needs it, we could also add a `depth` argument that limits the
    depth of traversal.

    Args:
        container: The container to iterate over
        iter_types: List of types that are treated as nested. If an objet of
            any other type is encountered that is not part of `iter_types` it
            is treated as a leaf.

    Yields:
        (path, value) tuples

    Examples:
        >>> list(nested_iter_items({'a': {'b': {'c': [1, 2, 42], 'd': 'e'}}}))
        [(('a', 'b', 'c', 0), 1), (('a', 'b', 'c', 1), 2), (('a', 'b', 'c', 2), 42), (('a', 'b', 'd'), 'e')]

        This can be useful for example if you have a list of nested elements:
        >>> c = [{'a': 1}, {'b': 2}, {'c': 2}]
        >>> for idx, element in enumerate(c):
        ...     for key, nested_element in element.items():
        ...         print(idx, key, nested_element)
        0 a 1
        1 b 2
        2 c 2

        >>> for (idx, key), nested_element in nested_iter_items(c):
        ...     print(idx, key, nested_element)
        0 a 1
        1 b 2
        2 c 2
    """
    try:
        items = container.items()
    except AttributeError:
        items = enumerate(container)
    for key, value in items:
        if isinstance(value, iter_types):
            for key_, value_ in nested_iter_items(value):
                yield (key,) + key_, value_
        else:
            yield (key,), value


class FlatView:
    def __init__(
            self,
            nested_container: Any,
            *,
            allow_partial_path: bool = False,
            sep: str = '.',
    ):
        """
        A view on a nested container that allows for access with paths. Useful
        if the nested container contains non-dict objects so that a `flatten`
        wouln't be reversible.

        Note:
            This view doesn't support `len` and `keys`, and thus is not a proper
            `Mapping`, because these are not easily accessible
            without traversing the whole nested structure.

        Examples:
            >>> d = {'a': 'b', 'c': {'d': {'e': 'f'}, 'g': [1, [2, 3], 4]}}
            >>> v = FlatView(d)
            >>> v['c.d.e']
            'f'
            >>> v.get('asdf', 'default')
            'default'
            >>> v['a'] = {'b': 'c'}
            >>> v['a.b']
            'c'
            >>> d['a']['b']
            'c'
            >>> v[('c', 'g', 1, 1)]
            3
            >>> v = FlatView(d, allow_partial_path=True)
            >>> v['a.b.c.d.e.f']
            'c'
            >>> list(v.items())
            [(('a', 'b'), 'c'), (('c', 'd', 'e'), 'f'), (('c', 'g', 0), 1), (('c', 'g', 1, 0), 2), (('c', 'g', 1, 1), 3), (('c', 'g', 2), 4)]
        """
        self.data = nested_container
        self.allow_partial_path = allow_partial_path
        self.sep = sep

    def get(
            self,
            item: Union[str, Tuple[Any, ...], None],
            default: Any = _NO_VALUE,
            *,
            sep: str = None
    ) -> Any:
        if sep is None:
            sep = self.sep
        return get_by_path(
            self.data, item, allow_early_stopping=self.allow_partial_path,
            default=default, sep=sep,
        )

    def items(self) -> Iterator[Tuple[Tuple[Any, ...], Any]]:
        """
        Yield the leaves and paths to the leaves, not all sub-containers.
        """
        yield from nested_iter_items(self.data)

    def keys(self) -> Iterator[Tuple[Any, ...]]:
        """
        Returns a generator over the paths to the leaves of the nested
        container. This is a sub-set of all possible keys that can be used with
        the `FlatView`.
        """
        # Should be slightly faster than a generator expression in most cases
        return map(operator.itemgetter(0), self.items())

    def values(self) -> Iterator[Any]:
        """
        Returns a generator over the leaf values of the nested container.
        """
        # Should be slightly faster than a generator expression in most cases
        return map(operator.itemgetter(1), self.items())

    def __len__(self):
        # This is faster than methods that don't store the contents like
        # sum([1 for _ in self.items()]), but since the content is anyways
        # already in the container, this should be fine
        return len(list(self.items()))

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        set_by_path(self.data, key, value)


def nested_any(x, fn: Callable = bool):
    """
    Checks if any value in the nested strucutre `x` evaluates to `True`

    Args:
        x: Nested structure to check
        fn: Function that is applied to every leaf before checking for truth.

    Returns:
        `True` if any leaf value in `x` evaluates to `True`.

    Examples:
        >>> nested_any([False, False, False])
        False
        >>> nested_any([True, False, False])
        True
        >>> nested_any({'a': False})
        False
        >>> nested_any({'a': False, 'b': True})
        True
        >>> nested_any([True, {'a': True, 'b': {'c': True}}, 1, 'true!'])
        True
        >>> nested_any([1, 2, 3, 4], fn=lambda x: x%2)
        True
    """
    for _, value in nested_iter_items(x):
        if fn(value):
            return True
    return False


def nested_all(x, fn: Callable = bool):
    """
    Checks if all values in the nested strucutre `x` evaluate to `True`

    Args:
        x: Nested structure to check
        fn: Function that is applied to every leaf before checking for truth.

    Returns:
        `True` if all leaf values in `x` evaluate to `True`.

    Examples:
        >>> nested_all([False, False, False])
        False
        >>> nested_all([True, True, True])
        True
        >>> nested_all([True, False, True])
        False
        >>> nested_all({'a': True})
        True
        >>> nested_all({'a': False, 'b': True})
        False
        >>> nested_all([True, {'a': True, 'b': {'c': True}}, 1, ''])
        False
        >>> nested_all([1, 2, 3, 4], fn=lambda x: x%2)
        False
        >>> nested_all([1, 3, 5, 7], fn=lambda x: x%2)
        True
    """
    # `all(x)` is the same as `not any([not x_ for x_ in x])`
    return not nested_any(x, fn=lambda x_: not fn(x_))
