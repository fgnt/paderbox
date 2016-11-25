import json
import os
import numpy as np
import bson
import datetime
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
    Numpy types will be converted to the equivalent python type for dumping the
    obj.

    :param obj: arbitary object that is json serializable, where numpy is
                allowed
    :param path:
    :param indent: see json.dump
    :param kwargs: see json.dump

    """
    if isinstance(path, (tuple, list)) and isinstance(path[0], str):
        path = os.path.join(path)

    if isinstance(path, str):
        with open(path, 'w') as f:
            json.dump(obj, f, cls=_Encoder, indent=indent, **kwargs)
    else:
        json.dump(obj, path, cls=_Encoder, indent=indent, **kwargs)


def load_json(*path_parts, **kwargs):
    """ Loads a json file and returns it as a dict

    :param path_parts: Json file name and possible parts of a path
    :param kwargs: see json.load
    :return: content of the json file
    """
    path = os.path.join(*path_parts)
    path = os.path.expanduser(path)

    with open(path) as fid:
        return json.load(fid, **kwargs)