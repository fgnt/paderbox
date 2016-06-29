import os
from warnings import warn
# from sacred.observers.mongo import *
import shutil
# from nt.utils.pynvml import *
import pandas as pd
from bson.objectid import ObjectId
from pymongo import MongoClient
from cached_property import cached_property


class SacredManager:
    def __init__(self, secret_file=None, uri=None):
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

    def _get_runs(self, database='sacred', prefix='default'):
        return self.client[database][prefix].runs

    def get_experiment_from_id(self, _id, database='sacred', prefix='default'):
        runs = self._get_runs(database, prefix)
        return runs.find_one({'_id': ObjectId(_id)})

    def get_config_from_id(self, _id, database='sacred', prefix='default'):
        return self.get_experiment_from_id(_id, database, prefix)['config']

    def get_info_from_id(self, _id, database='sacred', prefix='default'):
        return self.get_experiment_from_id(_id, database, prefix)['info']

    def delete_entry_by_id(
            self, _id, database='sacred', prefix='default', delete_dir=False
    ):
        runs = self._get_runs(database, prefix)
        if delete_dir:
            config = self.get_config_from_id(_id, database, prefix)
            try:
                data_dir = config[delete_dir]
            except KeyError:
                warn('{} is not part of the config. Cannot delete data dir.'.
                     format(delete_dir))
            else:
                shutil.rmtree(data_dir)
        delete_result = runs.delete_one({'_id': ObjectId(_id)})
        print(delete_result.raw_result)

    def get_data_frame(self, database='sacred', prefix='default', depth=2):
        def _make_data_frame(l):
            d = dict()

            def _f(indices, dic, i):
                # TODO: Don' know if this needs to be as cryptic as it is...
                for key in dic.keys():
                    val = dic[key]
                    new_indices = indices + (key,)
                    if type(val) is dict and len(new_indices) < depth:
                        _f(new_indices, val, i)
                    else:
                        new_indices = _expand_key(new_indices, depth)
                        if new_indices in d.keys():
                            d[new_indices][i] = val
                        else:
                            d[new_indices] = {i: val}

            for i, e in enumerate(l):
                _f(tuple(), e, i)
            return pd.DataFrame(d)

        runs = self._get_runs(database, prefix)
        list_of_dicts = list(runs.find())
        return _make_data_frame(list_of_dicts)


def _expand_key(key, length, expansion_key=''):
    # TODO: Please explain better, what this fn does. Parameters?
    if not type(key) is tuple:
        key = (key,)

    if len(key) < length:
        key = key + (expansion_key,) * (length - len(key))

    return key


def rename_columns(frame, map):
    # TODO: Please explain better, what this fn does. Parameters?
    # TODO: Why not df[('time', 'start')] = df['start_time']?
    depth = len(frame.columns.levels)
    for old_key in map.keys():
        new_key = map[old_key]
        frame[_expand_key(new_key, depth)] = frame[_expand_key(old_key, depth)]
        frame.drop(_expand_key(old_key, depth), 1, inplace=True)


def add_values(df, f):
    # TODO: Please explain better, what this fn does.
    depth = len(df.columns.levels)
    for i in df.index:
        try:
            (k, v) = f(df.loc[i].squeeze())
            df.loc[i, _expand_key(k, depth)] = v
        except:
            # TODO: Can you be more precise about the exception?
            pass


def filter_columns(frame, entries):
    # TODO: Unclear, why this works.
    def _in(lv, l):
        res = None
        for k in l:
            if res is None:
                res = (lv == k)
            else:
                res |= (lv == k)
        return res

    mask = None
    lv0 = frame.columns.get_level_values(0)
    lv1 = frame.columns.get_level_values(1)
    for k in entries.keys():
        v = entries[k]
        if type(v) is list:
            new_mask = (lv0 == k) & (_in(lv1, v))
        else:
            new_mask = (lv0 == k)

        if mask is None:
            mask = new_mask
        else:
            mask |= new_mask

    return frame.loc[:, mask]


def filter_rows(frame, entries):
    # TODO: How should filter_rows be used? Is there a great benefit to regular pandas selection in the notebook?
    # TODO: Why not fdf = fdf[(fdf['status'] == 'RUNNING')]?
    mask = None
    for v in entries:
        new_mask = frame[v[0]] == v[1]
        if mask is None:
            mask = new_mask
        else:
            mask &= new_mask

    return frame.loc[mask]


def print_columns(df, indent=0):
    try:
        for column in df.columns.levels[0]:
            print('    ' * indent + column)
            if not isinstance(df[column], pd.Series):
                print_columns(df[column], indent=indent+1)
    except AttributeError:
        for column in df.columns:
            print('    ' * indent + column)


def make_css_mark(mask, color_str='#FFFFFF'):
    css = ''
    for i in range(len(mask)):
        if mask.iloc[i]:
            css += 'tbody tr:nth-child(%d) {background-color: %s}\n' % (
                i+1, color_str
            )
    return css
