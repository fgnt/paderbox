import doctest


def assert_doctest_like_equal(
        want, got,
        options=('ELLIPSIS',),
        err_msg='',
):
    """

    >>> assert_doctest_like_equal('bla...', 'blab')
    >>> assert_doctest_like_equal('bla...', 'blub', err_msg='My err msg')
    Traceback (most recent call last):
    ...
    AssertionError: My err msg
    Expected:
        bla...
    Got:
        blub
    Differences (ndiff with -expected +actual):
        - bla...
        + blub

    """
    options = set(options) - {'REPORT_NDIFF', 'REPORT_CDIFF', 'REPORT_UDIFF'}

    msg = doctest_compare_str(
        want=want,
        got=got,
        options=options,
    )
    if msg is not None:
        diff_msg = doctest_compare_str(
            want=want,
            got=got,
            options=options | {'REPORT_NDIFF'},
        )
        raise AssertionError(
            err_msg + '\n' + msg + '\n' + diff_msg
        )


def doctest_compare_str(
        want, got,
        options=('ELLIPSIS',),
):
    """

    Allowed options: see doctest.OPTIONFLAGS_BY_NAME.keys()
     'DONT_ACCEPT_BLANKLINE',
     'DONT_ACCEPT_TRUE_FOR_1',
     'ELLIPSIS',
     'FAIL_FAST',
     'IGNORE_EXCEPTION_DETAIL',
     'NORMALIZE_WHITESPACE',
     'REPORT_CDIFF',
     'REPORT_NDIFF',
     'REPORT_ONLY_FIRST_FAILURE',
     'REPORT_UDIFF',
     'SKIP',

    >>> print(doctest_compare_str('bla...', 'blab'))
    None
    >>> print(doctest_compare_str('bla...', 'blub'))
    Expected:
        bla...
    Got:
        blub

    """
    optionflags = 0
    for o in options:
        optionflags |= doctest.OPTIONFLAGS_BY_NAME[o]

    checker = doctest.OutputChecker()
    checked = checker.check_output(want, got, optionflags)
    # print(checked, 'checked')
    if not checked:
        class Example:
            def __init__(self, want):
                self.want = want
        return checker.output_difference(
            Example(want + '\n'),
            got + '\n',
            optionflags,
        ).rstrip('\n')
