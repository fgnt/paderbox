import fnmatch

from IPython.display import display
import pandas as pd


def py_query(
        data: pd.DataFrame,
        query,
        *,
        use_pd_query=False,
        allow_empty_result=False,
        setup_code='',
        globals=None,
        return_selected_data=True,
):
    """
    Alternative: pd.DataFrame.query:
        supports a subset of this function, but is faster

    >>> df = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    >>> df
       a  b
    0  1  2
    1  3  4
    >>> py_query(df, 'a == 1')
       a  b
    0  1  2
    >>> py_query(df, 'a == 1', use_pd_query=True)
       a  b
    0  1  2
    >>> py_query(df, 'int(a) == 1')
       a  b
    0  1  2
    >>> py_query(df, ['int(a) == 1', 'b == 2'])
       a  b
    0  1  2
    >>> py_query(df, ['index == 1'])  # get second row
       a  b
    1  3  4
    >>> py_query(df, ['index == 1'], use_pd_query=True)
       a  b
    1  3  4

    To access column names that aren't valid python identifiers (e.g. the name
    contains a whitespace), you have to use the kwargs dictionary:
    >>> df = pd.DataFrame([{'a b': 1, 'b': 2}, {'a b': 3, 'b': 4}])
    >>> py_query(df, 'kwargs["a b"] == 1')
       a b  b
    0    1  2

    When you need a package function, you have to specify it in the globals
    dict. e.g.:
    >>> import numpy as np
    >>> df = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    >>> py_query(df, 'np.equal(a, 1)', globals={'np': np})
       a  b
    0  1  2

    Args:
        data: pandas.DataFrame
        query: str or list of str. If list of str the strings get join by a
            logical `and` to be a str. For examples see doctest.
             Note: Use index to get access to the index.
        use_pd_query:  Pandas query is much faster but limited
        allow_empty_result:
        setup_code: legacy argument, Superseded by the globals argument.
            Additional code which runs before the query conditions.
            You may use this for additional imports.
        globals: Specify some global names. Useful for imports. See doctest how
            to use it.
        return_selected_data: Whether to return the selection of the data or
            the selection indices.

    Returns:
        data[selection] if not return_selection else selection

    """
    if query is False:
        return data

    if query in [[], tuple(), '']:
        return data

    if isinstance(query, (list, tuple)):
        if len(query) == 1:
            query, = query
        else:
            query = ') and ('.join(query)
            query = f'({query})'
    else:
        assert isinstance(query, str)

    if use_pd_query is True:
        return data.query(query)
    elif use_pd_query == 'try':
        try:
            return data.query(query)
        except Exception:
            pass
    else:
        assert use_pd_query is False, use_pd_query

    keywords = ['index'] + list(data)

    def is_valid_variable_name(name):
        import ast
        # https://stackoverflow.com/a/36331242/5766934
        try:
            ast.parse('{} = None'.format(name))
            return True
        except (SyntaxError, ValueError, TypeError):
            return False

    keywords = [
        k
        for k in keywords
        if is_valid_variable_name(k)
    ] + ['**kwargs']

    d = {}
    code = f"""
def func({', '.join(keywords)}):
    {setup_code}
    try:
        return {query}
    except Exception:
        raise Exception('See above error message. Locals are:', locals())
"""

    if globals is None:
        globals = {}
    else:
        globals = globals.copy()
    try:
        exec(code, globals, d)
        func = d['func']
    except Exception as e:
        raise Exception(code) from e

    selection = data.apply(lambda row: func(row.name, **row), axis=1)
    assert allow_empty_result or len(selection) > 0, len(selection)
    if return_selected_data:
        return data[selection]
    else:
        return selection


def _unique_elements(pd_series):
    """
    Helper to get unique elements for non hashable sequences.

    >>> pd.Series([1, (2,)]).unique()
    array([1, (2,)], dtype=object)
    >>> s = pd.Series([1, 1, (1,), (1,), [1], [1], {1}, {1}, {5: {6: 7}}])
    >>> s
    0              1
    1              1
    2           (1,)
    3           (1,)
    4            [1]
    5            [1]
    6            {1}
    7            {1}
    8    {5: {6: 7}}
    dtype: object
    >>> _unique_elements(s)
    [1, (1,), [1], {1}, {5: {6: 7}}]
    >>> _unique_elements([1, 2, 3, 1, 3])
    [1, 2, 3]
    """

    try:
        return list(pd_series.unique())
    except Exception:
        pass

    mapping = {}
    import collections

    # map type to a unique identifier for this object
    o = collections.defaultdict(object)

    def make_hashable_inner(entry):
        try:
            hash(entry)  # TypeError: unhashable type: 'dict'
            return entry
        except TypeError:
            pass

        if isinstance(entry, list):
            new = tuple(entry)
            try:
                hash(new)  # TypeError: unhashable type: 'dict'
            except TypeError:
                new = tuple([make_hashable_inner(e) for e in entry])
        elif isinstance(entry, set):
            new = tuple(sorted(entry))
            try:
                hash(new)  # TypeError: unhashable type: 'dict'
            except TypeError:
                new = tuple([
                    make_hashable_inner(e)
                    for e in sorted(entry)
                ])
        elif isinstance(entry, dict):
            new = tuple(entry.items())
            try:
                hash(new)  # TypeError: unhashable type: 'dict'
            except TypeError:
                new = tuple(
                    {
                        make_hashable_inner(k): make_hashable_inner(v)
                        for k, v in entry.items()
                    }.items()
                )
        else:
            hash(entry)
            new = entry
        return new, o[type(entry)]

    def make_hashable(entry):
        try:
            hash(entry)  # TypeError: unhashable type: 'dict'
            return entry
        except TypeError:
            pass

        new = make_hashable_inner(entry)

        mapping[new] = entry

        return new

    ret = pd.Series(pd_series).apply(make_hashable).unique()

    return [mapping.get(r, r) for r in ret]


def squeeze_df(
        df: pd.DataFrame,
        query=None,
        drop=None,
        drop_glob=None,
        verbose=True,  # True or False as default?
        max_set_len=1,
        use_pd_query=False,
):
    """
    This function drops all unique columns of a dataset.
    With the argument drop and drop_glob further columns can be dropped.

    Args:
        df:
        query: See py_query function
        drop:
        drop_glob:
        verbose: If True print, print dropped unique columns.
        max_set_len: If verbose is True, print all values for columns with
                     <= max_set_len unique vales
        use_pd_query: See py_query function

    Returns:


    >>> df = pd.DataFrame([{'a': 3, 'b': 2}, {'a': 3, 'b': 4}, {'a': 3, 'b': 5}])
    >>> df
       a  b
    0  3  2
    1  3  4
    2  3  5
    >>> squeeze_df(df)
    a [3]
       b
    0  2
    1  4
    2  5
    >>> df = pd.DataFrame([{'a': [1], 'b': {1}}, {'a': [1], 'b': {1:1}}])
    >>> squeeze_df(df)
    a [[1]]
            b
    0     {1}
    1  {1: 1}

    """
    if query:
        df_tmp = py_query(df, query=query, use_pd_query=use_pd_query).copy()
    else:
        df_tmp = df.copy()

    if not drop:
        drop = []
    elif isinstance(drop, str):
        drop = [drop]
    else:
        drop = list(drop)

    for k, v in list(df_tmp.items()):
        unique = _unique_elements(v)
        if len(unique) == 1:
            drop.append(k)
            if verbose:
                print(k, unique)
        elif len(unique) <= max_set_len:
            if verbose:
                print(k, unique)

    if drop_glob:
        if isinstance(drop_glob, str):
            drop_glob = [drop_glob]
        for h in drop_glob:
            drop += fnmatch.filter(df_tmp.columns, h)

    df_tmp = df_tmp.drop(columns=drop)

    return df_tmp


def display_df(
        df,
        query=None,
        drop=None,
        drop_glob=None,
        max_set_len=1,
        verbose=True,
        use_pd_query=False,
):
    """
    Calls squeeze_df and give the result to IPython display.

    Args:
        df:
        query: See py_query function
        drop:
        drop_glob:
        verbose: If True print, print dropped unique columns.
        max_set_len: If verbose is True, print all values for columns with
                     <= max_set_len unique vales
        use_pd_query: See py_query function

    Returns:

    >>> df = pd.DataFrame([{'a': 3, 'b': 2}, {'a': 3, 'b': 4}, {'a': 3, 'b': 5}])
    >>> df
       a  b
    0  3  2
    1  3  4
    2  3  5
    >>> display_df(df)
    a [3]
       b
    0  2
    1  4
    2  5
    """
    display(squeeze_df(
        df=df,
        query=query,
        drop=drop,
        drop_glob=drop_glob,
        max_set_len=max_set_len,
        verbose=verbose,
        use_pd_query=use_pd_query,
    ))


def summary_df(df, query=None, max_unique_values=3):
    """
    Prints a summary for a dataframe, i.e. column name and unique values.

    Args:
        df:
        query: See py_query function
        max_unique_values:

    Returns:

    >>> df = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 3, 'b': 5}])
    >>> df
       a  b
    0  1  2
    1  3  4
    2  3  5
    >>> summary_df(df, max_unique_values=2)
    a [1, 3]
    b [2, 4, ...]

    """
    if query:
        df = py_query(df, query=query)

    for k, v in list(df.items()):
        unique = _unique_elements(v)
        if len(unique) <= max_unique_values:
            print(k, unique)
        else:
            class Dots:
                def __repr__(self):
                    return '...'
            print(k, unique[:max_unique_values] + [Dots()])
