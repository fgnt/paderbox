
def load_csv(file):
    """
    Load a csv file to python.

    This function should imitate pandas.read_csv, but should return instead
    of a pandas Dataframe a list of dicts.

    >>> from IPython.lib.pretty import pprint
    >>> file = '/net/db/wham_scripts/data/mix_2_spk_filenames_cv.csv'
    >>> pprint(load_csv(file))  # doctest: +ELLIPSIS
    [{'output_filename': '01to030v_0.76421_20ga010m_-0.76421.wav',
      's1_path': 'wsj0/si_tr_s/01t/01to030v.wav',
      's2_path': 'wsj0/si_tr_s/20g/20ga010m.wav'},
     {'output_filename': '40ec020o_1.3218_20ca010n_-1.3218.wav',
      's1_path': 'wsj0/si_tr_s/40e/40ec020o.wav',
      's2_path': 'wsj0/si_tr_s/20c/20ca010n.wav'},
     {'output_filename': '20lo010m_0.13154_01mc020p_-0.13154.wav',
      's1_path': 'wsj0/si_tr_s/20l/20lo010m.wav',
      's2_path': 'wsj0/si_tr_s/01m/01mc020p.wav'},
    ...

    >>> import pandas as pd
    >>> pd.read_csv(file)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                                   output_filename                        s1_path  \\
    0       01to030v_0.76421_20ga010m_-0.76421.wav  wsj0/si_tr_s/01t/01to030v.wav
    1         40ec020o_1.3218_20ca010n_-1.3218.wav  wsj0/si_tr_s/40e/40ec020o.wav
    2       20lo010m_0.13154_01mc020p_-0.13154.wav  wsj0/si_tr_s/20l/20lo010m.wav   
    ...

    """
    import csv

    def zip_same(a, b):
        assert len(a) == len(b), (len(a), len(b), a, b)
        return zip(a, b)

    with open(file, 'r') as fd:
        iterator = csv.reader(fd)
        header = next(iterator)
        return [
            dict(zip_same(header, row))
            for row in iterator
        ]
