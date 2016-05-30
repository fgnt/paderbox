import os
from pymongo import MongoClient
from bson.objectid import ObjectId


def get_sacred_uri_from_file(secret_file=None):
    """

    Store a mongodb uri in a file. I recommend `~/.sacred`. The file should
    contain i.e. `mongodb://user:password@131.234.222.24:10135`.

    Args:
        secret_file: Optional path to your sacred secret.

    Returns: Secret uri to connect to your db.

    """
    secret_file = '~/.sacred' if secret_file is None else secret_file
    secret_file = os.path.expanduser(secret_file)
    with open(secret_file, 'r') as f:
        uri = f.read().replace('\n', '')
    return uri


def get_config_from_id(_id, database='sacred', prefix='default',
                       secret_file=None):
    uri = get_sacred_uri_from_file(secret_file)
    uri += '/' if uri[-1] != '/' else ''
    uri += database
    client = MongoClient(uri)
    runs = client[database][prefix].runs
    experiment = runs.find_one({'_id': ObjectId(_id)})
    return experiment['config']
