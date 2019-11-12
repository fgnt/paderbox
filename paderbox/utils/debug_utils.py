import contextlib
import traceback
import sys
import pdb
import functools


__all__ = [
    'debug_on'
]


def is_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


class debug_on(contextlib.suppress):
    """
    Enters the debugger automatically if an exception occurs inside
    of the context.
    """
    def __init__(self, *exceptions, force_ipdb=True):
        if not exceptions:
            exceptions = [AssertionError]
        if len(exceptions) == 1 and isinstance(exceptions[0], (tuple, list)):
            exceptions = exceptions[0]
        super().__init__(*exceptions)
        self.force_ipdb = force_ipdb

    def __enter__(self):

        """
        >> with debug_on(AssertionError):
        ..     assert False
        """
        pass

    def post_mortem(self):
        try:
            traceback.print_exc()
            tb = sys.exc_info()[2]

            if self.force_ipdb or is_ipython():
                # https://github.com/gotcha/ipdb/blob/master/ipdb/__main__.py
                from IPython.terminal.interactiveshell import (
                    TerminalInteractiveShell
                )
                p = TerminalInteractiveShell().debugger_cls()
                p.botframe = sys._getframe().f_back  # I do not know why this
                # is nessesary, but without this hack it does not work
                p.interaction(None, tb)
            else:
                pdb.post_mortem(tb)
        except Exception as e:
            print('#'*40)
            print('debug_on does not alwais work with sacred. '
                  'Use -D for sacred applications')
            print('#'*40)
            raise e

    def __exit__(self, exctype, excinst, exctb):

        ret = super().__exit__(exctype, excinst, exctb)
        if ret is True:
            self.post_mortem()
            return False
        else:
            return ret

    def __call__(self, f):
        """
        >> @debug_on(AssertionError)
        .. def foo():
        ..     assert False
        >> foo()
        """
        # https://stackoverflow.com/a/12690039/5766934
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except self._exceptions:
                # traceback.print_exc()
                # pdb.post_mortem(sys.exc_info()[2])
                self.post_mortem()
                raise
        return wrapper
