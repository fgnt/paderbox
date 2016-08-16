import os


def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except FileNotFoundError:
        if path == '':
            pass


class change_directory:
    """Context manager for changing the current working directory

    http://stackoverflow.com/a/13197763/911441
    """
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)
