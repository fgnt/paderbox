import sh


def ccsinfo(host=None):
    """ Wrapper for ccsinfo.

    :param host: PC2 ssh host alias.
    """
    host = 'pc2' if host is None else host
    print(sh.ssh(host, 'ccsinfo', '-s', '--mine', '--fmt="%.R%.60N%.4w%P%D%v"'))


def ccskill(request_id, host=None):
    """ Wrapper for ccskill.

    :param request_id: Single request id or list of request ids.
    :param host: PC2 ssh host alias.
    """
    host = 'pc2' if host is None else host
    if isinstance(request_id, str):
        request_id = [request_id]
    print(sh.ssh(host, 'ccskill', request_id))
