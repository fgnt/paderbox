"""

Alternatives to AttrDict:

namedtuple:
>>> # https://docs.python.org/3/library/collections.html#collections.namedtuple
>>> from collections import namedtuple
>>> Point = namedtuple('Point', ['x', 'y'])
>>> p = Point(11, y=22)
>>> p.x, p[1]
(11, 22)

+ builtin
+- attributes are known
- tuple like, values are immutable
- no getitem access with keys

SimpleNamespace
>>> from types import SimpleNamespace
>>> p = SimpleNamespace()
>>> p.x = 3
>>> p
namespace(x=3)

+ builtin
+ dynamic append attributes
- no getitem access

AdDict
>>> from addict import Dict as AdDict
>>> p = AdDict(x=3)
>>> p.y = 2
>>> p['y'], p.x
(2, 3)

+ dict like
+ active developer behind addict
- external library
- recursive convert all dict's to AdDict
"""


# class AttrDict(dict):
#     """
#     http://stackoverflow.com/a/14620633
#     Has still a memory leak in python 3.5
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self


class AttrDict(dict):
    """

    A hack for a dictionary which is accessible with dot notation
    Source code: http://stackoverflow.com/a/5021467

    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("{} has no attribute '{}'"
                                 "".format(type(self), key))
        return self.__getitem__(key)
