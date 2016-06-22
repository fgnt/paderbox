import json
from warnings import warn
import os
from pymongo import MongoClient, ASCENDING
from bson.objectid import ObjectId
from sacred.observers.mongo import *
import shutil

import numpy as np
from nt.utils import nvidia_helper
import getpass
from nt.utils.pynvml import *
import pandas as pd

from IPython.core.display import display, HTML
from bson.objectid import ObjectId
from pymongo import MongoClient
from yattag import Doc


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
        uri = f.read().replace('\n', '').strip()
    return uri


def _get_runs(database='sacred', prefix='default', secret_file=None):
    uri = get_sacred_uri_from_file(secret_file)

    if not uri.endswith('/'):
        uri += '/'

    uri += database

    print(uri)
    client = MongoClient(uri)
    return client[database][prefix].runs


def get_experiment_from_id(_id, database='sacred', prefix='default',
                           secret_file=None):
    runs = _get_runs(database, prefix, secret_file)
    experiment = runs.find_one({'_id': ObjectId(_id)})
    return experiment


def get_config_from_id(_id, database='sacred', prefix='default',
                       secret_file=None):
    return get_experiment_from_id(_id, database, prefix, secret_file)['config']

def get_info_from_id(_id, database='sacred', prefix='default',
                       secret_file=None):
    return get_experiment_from_id(_id, database, prefix, secret_file)['info']


def delete_entry_by_id(_id, database='sacred', prefix='default',
                       secret_file=None, delete_dir=None):
    runs = _get_runs(database, prefix, secret_file)
    if delete_dir is not None:
        cfg = get_config_from_id(_id, database, prefix, secret_file)
        try:
            data_dir = cfg[delete_dir]
        except KeyError:
            warn('{} is not part of the config. Cannot delete data dir.'.format(
                delete_dir
            ))
        else:
            shutil.rmtree(data_dir)
    delete_result = runs.delete_one({'_id': ObjectId(_id)})
    print(delete_result.raw_result)


def print_overview_table(
        database='sacred', prefix='default', secret_file=None,
        constraints=None, callback_function=None, constraint_callback=None,
        config_blacklist=None, config_whitelist=None, show_hostinfo=True):
    constraints = {} if constraints is None else constraints

    runs = _get_runs(database, prefix, secret_file)

    list_of_dicts = list(runs.find(constraints))
    # list_of_dicts = list(runs.find(constraints).sort('start_time', ASCENDING))

    if len(list_of_dicts) == 0:
        print('No entries found for query')
        return

    doc, tag, text = Doc().tagtext()

    def _dict_to_cell(d):
        return '<br />'.join(
            json.dumps(d, indent=True, sort_keys=True).split('\n')[1:-1])

    with tag('small'):
        with tag('table', width='100%'):
            for row in list_of_dicts:
                if constraint_callback is None or constraint_callback(row):
                    if config_blacklist is not None:
                        row['config'] = {k: v for k, v in row['config'].items()
                                         if k not in config_blacklist}
                    if config_whitelist is not None:
                        row['config'] = {k: v for k, v in row['config'].items()
                                         if k in config_whitelist}
                    with tag('tr'):
                        with tag('td'):
                            text('id: {}'.format(row['_id']))
                            doc.stag('br')
                            text('heartbeat: {}'.format(
                                row['heartbeat'].strftime('%d.%m. %H:%M:%S')
                            ))
                            doc.stag('br')
                            text('name: {}'.format(row['experiment']['name']))
                            doc.stag('br')
                            text('status: {}'.format(row['status']))
                            doc.stag('br')
                            text('start_time: {}'.format(
                                row['start_time'].strftime('%d.%m. %H:%M:%S')
                            ))
                            doc.stag('br')
                            try:
                                text('stop_time: {}'.format(
                                    row['stop_time'].strftime('%d.%m. %H:%M:%S')
                                ))
                                doc.stag('br')
                                text('difference: {}'.format(
                                    str(
                                        row['stop_time'] - row['start_time']
                                    ).split('.')[0]
                                ))
                            except KeyError:
                                pass

                        with tag('td'):
                            doc.asis(_dict_to_cell(row['config']))
                        if show_hostinfo:
                            with tag('td'):
                                doc.asis(_dict_to_cell(row['host']))
                        if callback_function is not None:
                            with tag('td'):
                                try:
                                    doc.asis(callback_function(row))
                                except KeyError:
                                    pass

    display(HTML(doc.getvalue()))


def _expand_key(key, length, expansion_key=''):
    if not type(key) is tuple:
        key = (key,)

    if len(key) < length:
        key = key + (expansion_key,) * (length - len(key))

    return key


def rename_columns(frame, map):
    depth = len(frame.columns.levels)
    for old_key in map.keys():
        new_key = map[old_key]
        frame[_expand_key(new_key, depth)] = frame[_expand_key(old_key, depth)]
        frame.drop(_expand_key(old_key, depth), 1, inplace=True)


def add_values(df, f):
    depth = len(df.columns.levels)
    for i in df.index:
        try:
            (k, v) = f(df.loc[i].squeeze())
            df.loc[i, _expand_key(k, depth)] = v
        except:
            pass


def get_data_frame(database='sacred', prefix='default', secret_file=None,
                   depth=2):
    def _make_data_frame(l):
        d = dict()

        def _f(indices, dic, i):
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

    runs = _get_runs(database, prefix, secret_file)

    list_of_dicts = list(runs.find())

    frame = _make_data_frame(list_of_dicts)

    return frame


def filter_columns(frame, entries):
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
    mask = None
    for v in entries:
        new_mask = frame[v[0]] == v[1]
        if mask is None:
            mask = new_mask
        else:
            mask &= new_mask

    return frame.loc[mask]


class GPUMongoObserver(MongoObserver):
    @staticmethod
    def create(url='loclahost', db_name='sacred', prefix='default', **kwargs):
        """
            Does the same as MongoObserver.create but returns a GPUMongoObserver.
        """
        client = pymongo.MongoClient(url, **kwargs)
        database = client[db_name]
        for manipulator in SON_MANIPULATORS:
            database.add_son_manipulator(manipulator)
        runs_collection = database[prefix + '.runs']
        fs = gridfs.GridFS(database, collection=prefix)
        return GPUMongoObserver(runs_collection, fs)

    def started_event(self, ex_info, host_info, start_time, config, comment):
        """
            This Observer adds info about the GPUs of the Host-Computer to
            host_info. It's the only solution that works without modifying
            sacred source code.
            This should either be moved to get_host_info() in host_info.py or
            there should be a way to let the user add additional info (e.g.
            username). This could be achieved by modifying Ingredient and
            create_run in initialize.py.
        """
        try:
            gpu_info = nvidia_helper.get_info()
            gpu_list = nvidia_helper.get_gpu_list(print_error=False)
            host_info['gpu_count'] = gpu_info['device_count']
            host_info['gpu_info'] = {str(x['minor_number']): {
                'name': x['name'].decode(),
                'total_memory': str(x['memory']['total'] / 1048576) + 'Mib',
                'persistence_mode': x['persistence_mode'],
                'product_brand': x['product_brand'],
                'uuid': x['uuid'].decode(),
                'vbios_version': x['vbios_version'].decode()
            } for x in gpu_list}
        except NVMLError as e:
            host_info['gpu_count'] = 0
            host_info['gpu_info'] = None

        host_info['user'] = getpass.getuser()
        MongoObserver.started_event(self, ex_info, host_info, start_time,
                                    config,
                                    comment)
