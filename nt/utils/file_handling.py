import os
import errno

def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise