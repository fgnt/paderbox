import functools
import inspect
from typing import Callable


def partial_decorator(fn: Callable = None, *, nested: bool = True) -> Callable:
    """Allows to repeatedly call the function and add partial keyword args.

    Executes the function when all arguments are given as keyword arguments or
    a positional argument is passed. Similar to applying `functools.partial`
    (repeatedly if `nested=True`).

    This function went throuth multiple different ways of implementation, listed
     below for future reference:
     - Call `fn` when `*args` is present. This is simple and elegant, but only
        works if the first argument is not given as a keyword argument.
     - Get first argument name with `inspect.Signature` and call `fn` when
        either `*args` is present or the first argument is in `**kwargs`. This
        works but it is quite involved using Signature. Additionally, it only
        works when the first argument is given last, not for functions that have
        only kwargs / where their order doesn't matter.
    - Always call `fn` and return the partial nested_wrapper when an argument is
        missing (a `TypeError` is raised). While this seems elegant, it is not
        straightforward to detect if an argument is missing or unexpected and if
        the missing argument comes from the call of `fn` or from a function
        called within `fn`, especially if `fn` itself is decorated with another
        decorator that forwards `*args` and `**kwargs`.
    - Use `inspect.Signature.bind` to check if the signature is full or anything
        is missing. This, of course, only works when the function has a correct
        signature (i.e., doesn't just have `*args` and `**kwargs` or, if so,
        uses `functools.wraps`). This is the current implementation.

    Warnings:
        This decorator can cause bugs that are hard to find (e.g., a function
        not being called or having wrong arguments), especially when
        `nested=True`!

    Args:
        fn: The function to wrap
        nested: If `True`, the funtion can be called multiple times to add
            arguments successively. If `False`, only one call is permitted to
            set partial arguments and the second call always calls `fn` (or
            raises an exception if arguments are missing).

    Examples:
        >>> @partial_decorator
        ... def foo(data, a='a', b=42):
        ...    print(data, a, b)

        A call with all arguments calls foo directly
        >>> foo(123)
        123 a 42

        If other arguments are given, a partial call is performed
        >>> foo(a='b')(123)
        123 b 42

        Multiple partial calls can follow each other, if nested=True
        >>> foo(a='b')(b=43)(123)
        123 b 43

        The first argument can also be set by keyword
        >>> foo(a='b')(b=43)(data=123)
        123 b 43

        The partial kwargs are not saved in the function, but reset for each
        new chain of calls
        >>> foo(42)
        42 a 42

        Multiple partial calls can be disabled by setting nested=False
        >>> @partial_decorator(nested=False)
        ... def bar(a, b, c):
        ...     print(a, b)
        >>> bar(b=1)(c=2)
        Traceback (most recent call last):
          ...
        TypeError: bar() missing 1 required positional argument: 'a'

        If there are multiple positional arguments, these can be specified in
        any order when partial calls are chained
        >>> @partial_decorator
        ... def foobar(a, b, c):
        ...     print(a, b, c)
        >>> foobar(b=1)(a=2)(c=4)
        2 1 4

        Other wrong uses that raise a TypeError are raised as expected
        >>> foobar(1, 2, 3, 4)
        Traceback (most recent call last):
          ...
        TypeError: too many positional arguments
        >>> foobar(1, 2, 3, d=4)
        Traceback (most recent call last):
          ...
        TypeError: got an unexpected keyword argument 'd'

        Unexpected argument errors are raised early. This makes spotting the
        place where the unexpected argument was added easier.
        >>> foobar(d=4)
        Traceback (most recent call last):
          ...
        TypeError: got an unexpected keyword argument 'd'
        >>> foobar(a=1)(d=4)
        Traceback (most recent call last):
          ...
        TypeError: got an unexpected keyword argument 'd'

    """
    if fn is None:
        return functools.partial(partial_decorator, nested=nested)

    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def nested_wrapper(*args, **kwargs):
        try:
            # Check if all arguments are present
            signature.bind(*args, **kwargs)
        except TypeError as e:
            if args:
                # It doesn't make sense to make a partial call with args
                raise

            # Check if the TypeError was a missing argument or any other wrong
            # use. Re-raise exception if it is not caused by a missing argument
            if 'missing a required argument' not in e.args[0]:
                raise

            # Check for other errors, e.g., unexpected args early
            try:
                signature.bind_partial(*args, **kwargs)
            except TypeError as e:
                # The initial exception is not informative here
                raise e from None

            if nested:
                return functools.partial(nested_wrapper, **kwargs)
            else:
                return functools.partial(fn, **kwargs)
        else:
            return fn(*args, **kwargs)

    return nested_wrapper
