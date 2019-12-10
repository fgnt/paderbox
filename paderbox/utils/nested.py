import collections
import itertools


def flatten(d, sep='.', flat_type=dict):
    """Flatten a nested dict using a specific separator.

    :param sep: When None, return dict with tuple keys (guaranties inversion of
                flatten) else join the keys with sep
    :param flat_type: Allow other mappings instead of flat_type to be
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


def deflatten(d, sep='.', maxdepth=-1):
    """Build a nested dict from a flat dict respecting a separator.

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
            sub_dict = sub_dict[sub_key]
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
    The last dict has the highest priority when allow_update is True.
    When allow_update is False, it is assumed, that no values gets overwritten.

    When inplace is True, the default_dict is manypulated inplace. This is
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

    """
    if len(update_dicts) == 0:
        if inplace:
            return default_dict
        else:
            return default_dict.__class__(default_dict)

    dicts = [default_dict, *update_dicts]

    def get_value_for_key(key):
        values = [
            d[key]
            for d in dicts
            if key in d.keys()
        ]
        if isinstance(values[-1], collections.abc.Mapping):
            return nested_merge(*[
                v for v in values if isinstance(v, collections.abc.Mapping)
            ], allow_update=allow_update, inplace=inplace)
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

    if inplace:
        for k in keys:
            default_dict[k] = get_value_for_key(k)
        return default_dict
    else:
        return default_dict.__class__({
            k: get_value_for_key(k)
            for k in keys
        })


def nested_op(
        func,
        arg1, *args,
        broadcast=False,
        handle_dataclass=False,
        keep_type=True,
        mapping_type=collections.abc.Mapping,
        sequence_type=(tuple, list),

):
    """This function is `nested_map` with a fancy name.

    Applies the function "func" to the leafs of the nested data structures.
    This is similar to the map function that applies the function the the
    elements of an iterable input (e.g. list).

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
                broadcast=broadcast,
                mapping_type=mapping_type,
                sequence_type=sequence_type,
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
                broadcast=broadcast,
                mapping_type=mapping_type,
                sequence_type=sequence_type,
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
                    broadcast=broadcast,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
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