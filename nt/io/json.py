import json
import numpy as np
import bson
import datetime
from pathlib import Path

from chainer import Variable

# http://stackoverflow.com/a/27050186
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Variable):
            return obj.num.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bson.objectid.ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            return super().default(obj)


def dump_json(obj, path, *, indent=2, **kwargs):
    """
    Numpy types will be converted to the equivalent Python type for dumping the
    object.

    :param obj: Arbitrary object that is JSON serializable,
        where Numpy is allowed.
    :param path: String or ``pathlib.Path`` object.
    :param indent: See ``json.dump()``.
    :param kwargs: See ``json.dump()``.

    """
    assert isinstance(path, (str, Path))
    path = Path(path).expanduser()

    with path.open('w') as f:
        json.dump(obj, f, cls=_Encoder, indent=indent, sort_keys=True, **kwargs)


def load_json(path, **kwargs):
    """ Loads a JSON file and returns it as a dict.

    :param path: String or ``pathlib.Path`` object.
    :param kwargs: See ``json.dump()``.
    :return: Content of the JSON file.
    """
    assert isinstance(path, (str, Path))
    path = Path(path).expanduser()

    with path.open() as fid:
        return json.load(fid, **kwargs)
