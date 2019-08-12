#!/usr/bin/env python

import os
import pickle
import h5py
import json
import click
import numpy
import math
import magic
from functools import partial
from contextlib import contextmanager

suffixes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']


def human_size(nbytes):
  rank = int((math.log10(nbytes)) / 3)
  rank = min(rank, len(suffixes) - 1)
  human = nbytes / (1024.0 ** rank)
  f = ('%.2f' % human).rstrip('0').rstrip('.')
  return '%s %s' % (f, suffixes[rank])


def echo(prefix, file_name):
    click.echo(click.style('{} {}, {}'.format(
        prefix, file_name, human_size(os.path.getsize(file_name))), fg='blue'))


def hdf5_to_dict(f):
    if isinstance(f, (h5py._hl.files.File, h5py._hl.group.Group)):
        return {name: hdf5_to_dict(f[name]) for name in f}
    else:
        return f[:]


# http://stackoverflow.com/a/27050186
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.complex):
            # ToDo: json serilize complex numbers in a better way
            # return {"Complex": [obj.real, obj.imag]}
            return '{: .3g} + j {:.3g}'.format(obj.real, obj.imag)
            #return str(obj) #{'complex': (obj.real, obj.imag)}
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, h5py._hl.files.File):
            return hdf5_to_dict(obj)
        else:
            return super(MyEncoder, self).default(obj)

@contextmanager
def load(file, mime):
    if mime == 'pkl':
        with open(file, 'rb') as f:
            data = pickle.load(f)
        yield data
    elif mime == 'hdf5':
        with h5py.File(file) as f:
            yield f
    else:
        raise ValueError('mime: '.format(mime))


@click.command()
@click.argument('file', type=click.Path(exists=True), )
@click.option('--mime', type=str, default=None)
def main(file, mime):
    mime_magic = magic.Magic(mime=True)

    if mime is None:
        mime_str = mime_magic.from_file(file)
        mime_map = {
            'application/octet-stream': 'pkl',
            'application/x-hdf': 'hdf5',
        }
        try:
            mime = mime_map[mime_str]
        except KeyError:
            print('mime_str: ', mime_str)
            raise

    assert mime in ('pkl', 'hdf5')

    echo('Load', file)
    with load(file, mime) as data:
        file_json = os.path.splitext(file)[0] + '.json'

        with open(file_json, 'w') as f:
            json.dump(data, f, cls=MyEncoder, indent=2, separators=(',', ':'))
        echo('Dump', file_json)

if __name__ == '__main__':
    main()