import os
from warnings import warn
import shutil
import pandas as pd
from bson.objectid import ObjectId
from pymongo import MongoClient
from cached_property import cached_property
from nt.utils.pandas_helper import colorize_and_display_dataframe, \
    filter_columns, make_css_mark, print_columns  # for backward compatibility
from nt.utils.pandas_helper import set_values


class SacredManager:
    def __init__(self, secret_file=None, uri=None, database=None, prefix=None):
        """ Create a SacredManager which provides info about sacred db.

        Args:
            secret_file: Path to your sacred secret, i.e. `~/.sacred`.
            uri: Alternatively, provide a uri i.e.
                `mongodb://user:password@131.234.222.24:10135`.
        """
        if uri is None:
            secret_file = '~/.sacred' if secret_file is None else secret_file
            secret_file = os.path.expanduser(secret_file)
            with open(secret_file, 'r') as f:
                self.uri = f.read().replace('\n', '').strip()
        else:
            assert secret_file is None, 'Provide either secret_file or uri.'
            self.uri = uri

        # I decided to not set the properties if None to get good error messages
        if database is not None:
            self.database = database
        if prefix is not None:
            self.prefix = prefix

    @cached_property
    def client(self):
        return MongoClient(self.uri)

    def print_database_and_collection_names(self):
        """ Print overview about MongoDB server with databases and collections.

        Only works, when there is no authentication required.
        """
        # TODO: Improve behavior for databases with authentication.
        for database in self.client.database_names():
            print(database)
            for collection in self.client[database].collection_names():
                print('    ' + collection)

    def _get_runs(self):
        return self.client[self.database][self.prefix].runs

    def get_experiment_from_id(self, _id):
        runs = self._get_runs()
        return runs.find_one({'_id': ObjectId(_id)})

    def get_config_from_id(self, _id):
        return self.get_experiment_from_id(_id)['config']

    def get_info_from_id(self, _id):
        return self.get_experiment_from_id(_id)['info']

    def delete_entry_by_id(self, _id, delete_dir=False):
        runs = self._get_runs()
        if delete_dir:
            config = self.get_config_from_id(_id)
            try:
                data_dir = config[delete_dir]
            except KeyError:
                warn('{} is not part of the config. Cannot delete data dir.'.
                     format(delete_dir))
            else:
                shutil.rmtree(data_dir)
        delete_result = runs.delete_one({'_id': ObjectId(_id)})
        print(delete_result.raw_result)

    def set_db_values(self, _id, values):
        set_values(self._get_runs(), _id, values)

    def get_data_frame(self, depth=2):
        """
        Creates a pandas DataFrame with a MultiIndex with depth count of levels
        out of entries found in database with prefix prefix. Keys for the
        MultiIndex are taken from the keys of nested dicts in the database
        entries.

        :param database:
        :param prefix:
        :param depth: length of the keys of the MultiIndex of the DataFrame
        :return:
        """
        runs = self._get_runs()
        list_of_dicts = list(runs.find())

        d = dict()

        # TODO: Don't know if this needs to be as cryptic as it is...
        def _multiindex_key_dict(indices, dic, i):
            """Creates a dict with Tuples of length depth as keys so that
            the constructor of DataFrame creates a MultiIndex out of it.
            For recursive calls, indices specifies a prefix of all keys
            created in this call. dic is a dictionary (row) whose values
            should be added to dict d and i is the row index."""
            for key in dic.keys():
                val = dic[key]
                new_indices = indices + (key,)
                if type(val) is dict and len(new_indices) < depth:
                    # recursively go deeper into nested dicts
                    _multiindex_key_dict(new_indices, val, i)
                else:
                    # reached end of nested dicts or desired depth.
                    # create key of given length
                    new_indices = _expand_key(new_indices, depth)

                    # add value from dict
                    if new_indices in d.keys():
                        d[new_indices][i] = val
                    else:
                        d[new_indices] = {i: val}

        for i, e in enumerate(list_of_dicts):
            _multiindex_key_dict(tuple(), e, i)
        return pd.DataFrame(d)


def _expand_key(key, length, expansion_key=''):
    """Creates a Key for a MultiIndex (tuple) of length length and fills
    missing entries with expansion_key. This is needed to create a DataFrame
    with a MultiIndex of given depth."""
    if not type(key) is tuple:
        key = (key,)

    if len(key) < length:
        key = key + (expansion_key,) * (length - len(key))

    return key
