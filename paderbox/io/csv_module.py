
def load_csv(file):
    """
    Load a csv file to python.

    This function should imitate pandas.read_csv, but should return instead
    of a pandas Dataframe a list of dicts.


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

    """
    import csv, io

    if not isinstance(file, io.IOBase):
        with open(file, 'r') as fd:
            return load_csv(fd)

    def zip_same(a, b):
        assert len(a) == len(b), (len(a), len(b), a, b)
        return zip(a, b)

    # with open(file, 'r') as fd:
    iterator = csv.reader(file)
    header = next(iterator)
    return [
        dict(zip_same(header, row))
        for row in iterator
    ]

def loads_csv(content):
    import io
    return load_csv(io.StringIO(content))

