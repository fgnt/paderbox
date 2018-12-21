def flatten(d, sep='.', flat_type=dict):
    """Flatten a nested dict using a specific separator.

    :param sep: When None, return dict with tuple keys (guaranties inversion of
                flatten) else join the keys with sep
    :param flat_type: Allow other mappings instead of flat_type to be
                flattened, e.g. using an isinstance check.

    import collections
    flat_type=collections.MutableMapping

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


def deflatten(d, sep='.'):
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
    >>> d = flatten(d_in, sep='_')
    >>> for k, v in d.items(): print(k, v)
    a 1
    c.a 2
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
            tuple(k.split(sep)): v for k, v in d.items()
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
    assert isinstance(update, type(orig))
    if isinstance(orig, list):
        for i, value in enumerate(update):
            if isinstance(value, (dict, list)) \
                    and i < len(orig) and isinstance(orig[i], type(value)):
                nested_update(orig[i], value)
            elif i < len(orig):
                orig[i] = value
            else:
                assert i == len(orig)
                orig.append(value)
    elif isinstance(orig, dict):
        for key, value in update.items():
            if isinstance(value, (dict, list)) \
                    and key in orig and isinstance(orig[key], type(value)):
                nested_update(orig[key], value)
            else:
                orig[key] = value


def nested_op(func, arg1, *args, broadcast=False):
    """
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=[1], b=dict(c=4)), dict(a=[0], b=dict(c=1)))
    {'a': [1], 'b': {'c': 7}}
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3]))
    Traceback (most recent call last):
    ...
    AssertionError: ([1, 3],)
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3]), broadcast=True)
    {'a': 1, 'b': {'c': [4, 10]}}

    :param func:
    :param arg1:
    :param args:
    :param broadcast:
    :return:
    """
    if isinstance(arg1, dict):
        if not broadcast:
            assert all(
                [isinstance(arg, dict) and arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        else:
            assert all(
                [not isinstance(arg, dict) or arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        keys = arg1.keys()
        return arg1.__class__({
            key: nested_op(
                func,
                arg1[key],
                *[arg[key] if isinstance(arg, dict) else arg
                  for arg in args],
                broadcast=broadcast
            )
            for key in keys
        })
    if isinstance(arg1, (list, tuple)):
        if not broadcast:
            assert all([
                isinstance(arg, (list, tuple)) and len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not isinstance(arg, (list, tuple)) or len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        return arg1.__class__([
            nested_op(
                func,
                arg1[j],
                *[arg[j] if isinstance(arg, (list, tuple)) else arg
                  for arg in args],
                broadcast=broadcast
            )
            for j in range(len(arg1))]
        )
    return func(arg1, *args)


def squeeze_nested(orig):
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