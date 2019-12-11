class DispatchError(KeyError):
    def __str__(self):
        if len(self.args) == 2:
            item, keys = self.args
            import difflib
            # Suggestions are sorted by their similarity.
            suggestions = difflib.get_close_matches(
                item, keys, cutoff=0, n=100
            )
            return f'Invalid option {item!r}.\n' \
                   f'Close matches: {suggestions!r}.'
        else:
            return super().__str__()


class Dispatcher(dict):
    """
    Is basically a dict with a better error message on key error.
    >>> from paderbox.utils.mapping import Dispatcher
    >>> d = Dispatcher(abc=1, bcd=2)
    >>> d['acd']  #doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    paderbox.utils.mapping.DispatchError: Invalid option 'acd'.
    Close matches: ['bcd', 'abc'].
    """

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError as e:
            raise DispatchError(item, self.keys()) from e
