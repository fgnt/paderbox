import json
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from sacred.observers import MongoObserver
from nt.utils import nvidia_helper
import getpass
from nt.utils.pynvml import *

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


def delete_entry_by_id(_id, database='sacred', prefix='default',
                       secret_file=None):
    uri = get_sacred_uri_from_file(secret_file)
    uri += '/' if uri[-1] != '/' else ''
    uri += database
    client = MongoClient(uri)
    runs = client[database][prefix].runs
    delete_result = runs.delete_one({'_id': ObjectId(_id)})
    print(delete_result.raw_result)


def print_overview_table(
        database='sacred', prefix='default', secret_file=None,
        constraints=None, callback_function=None, constraint_callback=None,
        config_blacklist=None, config_whitelist=None
):
    constraints = {} if constraints is None else constraints

    uri = get_sacred_uri_from_file(secret_file)
    uri += '/' if uri[-1] != '/' else ''
    uri += database
    client = MongoClient(uri)
    runs = client[database][prefix].runs

    list_of_dicts = list(runs.find(constraints))

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
                        with tag('td'):
                            doc.asis(_dict_to_cell(row['host']))
                        if callback_function is not None:
                            with tag('td'):
                                try:
                                    doc.asis(callback_function(row))
                                except KeyError:
                                    pass

    display(HTML(doc.getvalue()))


class GPUMongoObserver(MongoObserver):
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
            gpu_list = nvidia_helper.get_gpu_list()
            host_info['gpu_count'] = gpu_info['device_count']
            host_info['gpu_info'] = {str(x['minor_number']): {
                'name': x['name'].decode(),
                'total_memory': str(x['memory']['total'] / 1048576) + 'Mib',
                'persistence_mode': str(x['persistence_mode']),
                'product_brand': str(x['product_brand']),
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
