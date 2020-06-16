from typing import Optional, List


def load_csv(
        file,
        *,
        fieldnames: Optional[List[str]] = None,
        dialect: str = "excel",
        sniffer: bool = False,
        sniffer_sample_length: int = 1024,
) -> List[dict]:
    r"""
    Load a csv file to python.

    This function should imitate pandas.read_csv, but should return instead
    of a pandas Dataframe a list of dicts.

    Args:
        file:
        fieldnames:
            When a csv has no header, you can manually specify the fieldnames.
            Note, when fieldnames is given, it is assumed, that the csv has no
            header.
        dialect:
            e.g. 'excel', 'excel-tab' and 'unix'.
            See python documentation for standard library csv for more details.
        sniffer:
            Whether to detect the dialect (i.e. delimiter and line separator)
            When True, ignore dialect.
        sniffer_sample_length:

    Returns:
        list of dicts

    >>> from IPython.lib.pretty import pprint
    >>> from pathlib import Path
    >>> import sklearn
    >>> file = Path(sklearn.__file__).parent / 'datasets/data/iris.csv'
    >>> # file = '/net/db/wham_scripts/data/mix_2_spk_filenames_cv.csv'
    >>> pprint(load_csv(file))  # doctest: +ELLIPSIS
    [{'150': '5.1',
      '4': '3.5',
      'setosa': '1.4',
      'versicolor': '0.2',
      'virginica': '0'},
     {'150': '4.9',
      '4': '3.0',
      'setosa': '1.4',
      'versicolor': '0.2',
      'virginica': '0'},
    ...

    >>> import pandas as pd
    >>> pd.read_csv(file)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         150    4  setosa  versicolor  virginica
    0    5.1  3.5     1.4         0.2          0
    1    4.9  3.0     1.4         0.2          0
    ...

    When the a csv file does not use the standard `dialect` (i.e. 'excel'),
    you can use the sniffer flag to detect the `dialect`
    >>> content = 'a b c\n1 2 3\n4 5 6'
    >>> loads_csv(content)
    [{'a b c': '1 2 3'}, {'a b c': '4 5 6'}]
    >>> loads_csv(content, sniffer=True)
    [{'a': '1', 'b': '2', 'c': '3'}, {'a': '4', 'b': '5', 'c': '6'}]
    >>> loads_csv(content, fieldnames=['d', 'e', 'f'], sniffer=True)
    [{'d': 'a', 'e': 'b', 'f': 'c'}, {'d': '1', 'e': '2', 'f': '3'}, {'d': '4', 'e': '5', 'f': '6'}]
    >>> content = 'a\tb\tc\n1\t2\t3\n4\t5\t6'
    >>> loads_csv(content, sniffer=True)
    [{'a': '1', 'b': '2', 'c': '3'}, {'a': '4', 'b': '5', 'c': '6'}]

    """
    import csv, io, contextlib

    with contextlib.ExitStack() as exit_stack:
        if not isinstance(file, io.TextIOBase):
            file = exit_stack.enter_context(open(file, 'r'))

        if sniffer:
            # https://docs.python.org/3/library/csv.html#csv.Sniffer
            dialect = csv.Sniffer().sniff(file.read(sniffer_sample_length))
            file.seek(0)

        iterator = csv.DictReader(file, fieldnames=fieldnames, dialect=dialect)

        # Remove ordereddict, since CPython 3.6 and Python 3.7 no longer
        # necessary.
        return list(map(dict, iterator))


def loads_csv(
        content,
        *,
        fieldnames: Optional[List[str]] = None,
        dialect: str = "excel",
        sniffer: bool = False,
        sniffer_sample_length: int = 1024,
):
    import io
    return load_csv(
        io.StringIO(content),
        fieldnames=fieldnames,
        dialect=dialect,
        sniffer=sniffer,
        sniffer_sample_length=sniffer_sample_length,
    )
