import functools
import inspect
from typing import Callable


def partial_decorator(
        fn: Callable = None,
        *,
        chain: bool = False,
        requires_partial_call: bool = False,
) -> Callable:
    """Allows to repeatedly call the function and add partial keyword args.

    Executes the function when all arguments are given as keyword arguments or
    a positional argument is passed. Similar to applying `functools.partial`
    (repeatedly if `nested=True`).
    
    Positional arguments are not allowed for partial calls, i.e., calls that 
    don't actually call `fn`. This is for multiple reasons, even if 
    `requires_partial_call=True`:
     - It is not possible (or, very hard) to detect if a function has * or ** 
        arguments and then unclear how to merge them with keyword arguments
     - It is unclear how to merge positional arguments if calls with positional 
        arguments are chained (e.g., if one of three positional argumetns is 
        specified in the first partial call, should the positional arguments of 
        the next partial call be appended or overwrite? What if the first call 
        specifies three positional arguments and the second call only one with 
        an *args argument?)
     - It could be confusing

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
        chain: If `True`, the funtion can be called multiple times to add
            arguments successively. If `False`, only one call is permitted to
            set partial arguments and the second call always calls `fn` (or
            raises an exception if arguments are missing).
        requires_partial_call: If `True`, a nested call is required, so directly
            calling the function with all parameteres will not call `fn`.

    Examples:
        >>> @partial_decorator
        ... def foo(a, b, c=42):
        ...     print(a, b, c)

        A call with all (required) arguments calls foo directly
        >>> foo(1, 2)
        1 2 42

        If other arguments are given, a partial call is performed
        >>> foo(b='b')(123)
        123 b 42

        The first argument can also be set by keyword
        >>> foo(c='b', b=43)(a=123)
        123 43 b

        If chain is false (default), only one partial call is allowed
        >>> foo(b=1)(c=2)
        Traceback (most recent call last):
          ...
        TypeError: foo() missing 1 required positional argument: 'a'

        Chained partial calls can be activated by setting chain=True
        >>> @partial_decorator(chain=True)
        ... def bar(data, a='a', b=42):
        ...    print(data, a, b)

        Multiple partial calls can follow each other, if nested=True
        >>> bar(a='b')(b=43)(123)
        123 b 43

        The same argument can be overwritten multiple times
        >>> bar(a='b')(a='c')(123)
        123 c 42

        The partial kwargs are not saved in the function, but reset for each
        new chain of calls
        >>> bar(42)
        42 a 42

        If there are multiple positional arguments, these can be specified in
        any order when partial calls are chained
        >>> @partial_decorator(chain=True)
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

        With requires_partial_call, the function always has to be called at least
        twicerequires_partial_call
        >>> @partial_decorator(requires_partial_call=True, chain=True)
        ... def baz(a, b):
        ...     print(a, b)
        >>> baz(a=1, b=2)()
        1 2

        Positional arguments are still not allowed for the partial call, even if
        all of them are given
        >>> baz(1, 2)()
        Traceback (most recent call last):
          ...
        RuntimeError: Can't make a partial call with positional arguments (you set requires_partial_call=True).
        >>> baz(a=1)(a=2)(b=3)
        2 3

        Check keyword arguments and variable numbers of arguments
        >>> @partial_decorator
        ... def buzz(a, *args, **kwargs):
        ...     print(a, args, kwargs)
        >>> buzz(keyword1=42, keyword2=123)(1, 2, 3, 4, 5)
        1 (2, 3, 4, 5) {'keyword1': 42, 'keyword2': 123}
    """
    if fn is None:
        return functools.partial(
            partial_decorator,
            chain=chain,
            requires_partial_call=requires_partial_call
        )

    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def partial_wrapper(
            *args,
            __requires_partial_call=requires_partial_call,
            **kwargs
    ):
        """
        Args:
            __requires_partial_call: Has to be passed to this function instead
                of modifying `requires_partial_call` directly to not cause any
                side effects.
        """

        try:
            # Check if all arguments are present
            bound_args = signature.bind(*args, **kwargs)
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

            if chain:
                return functools.partial(
                    partial_wrapper, **kwargs, __requires_partial_call=False
                )
            else:
                return functools.partial(fn, **kwargs)
        else:
            if __requires_partial_call:
                if args:
                    # If we find a way to detect if fn has an *args argument and
                    # if it is filled, we can also allow positional arguments
                    # here as long as they don't end up in *args.
                    raise RuntimeError(
                        f'Can\'t make a partial call with positional arguments '
                        f'(you set requires_partial_call=True).'
                    )
                return functools.partial(
                    partial_wrapper, **kwargs,
                    __requires_partial_call=False
                )
            return fn(*args, **kwargs)

    return partial_wrapper
