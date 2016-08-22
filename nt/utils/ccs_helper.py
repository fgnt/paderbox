import sh
import pandas as pd
from io import StringIO


def ccsinfo(host=None):
    """ Wrapper for ccsinfo.

    :param host: PC2 ssh host alias.
    """
    host = 'pc2' if host is None else host
    ret = sh.ssh(host, 'ccsinfo', '-s', '--mine', '--fmt="%.R%.60N%.4w%P%D%v"',
                 '--raw')
    return pd.read_csv(StringIO(ret.stdout.decode('utf-8')),
                       delim_whitespace=True,
                       names=['id', 'name', 'status', 'start_time',
                              'time_limit', 'runtime'])


def ccskill(request_id, host=None):
    """ Wrapper for ccskill.

    :param request_id: Single request id or list of request ids.
    :param host: PC2 ssh host alias.
    """
    host = 'pc2' if host is None else host
    if isinstance(request_id, str):
        request_id = [request_id]
    print(sh.ssh(host, 'ccskill', request_id))
