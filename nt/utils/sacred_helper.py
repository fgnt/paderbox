import os
from pymongo import MongoClient
from bson.objectid import ObjectId

from IPython.core.display import display, HTML
from yattag import Doc
import json


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


def delete_entry_by_id(_id, database='sacred', prefix='default', secret_file=None):
    uri = get_sacred_uri_from_file(secret_file)
    uri += '/' if uri[-1] != '/' else ''
    uri += database
    client = MongoClient(uri)
    runs = client[database][prefix].runs
    delete_result = runs.delete_one({'_id': ObjectId(_id)})
    print(delete_result.raw_result)


def print_overview_table(
        database='sacred', prefix='default', secret_file=None,
        constraints=None, callback_function=None, constraint_callback=None
):
    constraints = {} if constraints is None else constraints

    uri = get_sacred_uri_from_file(secret_file)
    uri += '/' if uri[-1] != '/' else ''
    uri += database
    client = MongoClient(uri)
    runs = client[database][prefix].runs

    list_of_dicts = list(runs.find(constraints))

    doc, tag, text = Doc().tagtext()

    with tag('small'):
        with tag('table', width='100%'):
            for row in list_of_dicts:
                if constraint_callback is None or constraint_callback(row):
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
                            doc.asis(json.dumps(
                                row['config'],
                                indent=True, sort_keys=True
                            ).replace('\n', '<br />'))
                        with tag('td'):
                            doc.asis(json.dumps(
                                row['host'],
                                indent=True, sort_keys=True
                            ).replace('\n', '<br />'))
                        if callback_function is not None:
                            with tag('td'):
                                try:
                                    doc.asis(callback_function(row))
                                except KeyError:
                                    pass

    display(HTML(doc.getvalue()))
