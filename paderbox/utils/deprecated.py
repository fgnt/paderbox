import functools
import inspect
import warnings


# NOTE(kgriffs): We don't want our deprecations to be ignored by default,
# so create our own type.
class DeprecatedWarning(UserWarning):
    pass


def deprecated(instructions):
    """
    Original: https://gist.github.com/kgriffs/8202106

    Flags a method as deprecated.
    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        already_warned = False

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal already_warned
            if not already_warned:
                already_warned = True
                message = 'Call to deprecated function {} ({}). {}'.format(
                    func.__qualname__,
                    inspect.getfile(func),
                    instructions)

                # fileno = func.__code__.co_firstlineno if isinstance(func, types.FunctionType) else -1
                #
                # warnings.warn_explicit(message,
                #                        category=DeprecatedWarning,
                #                        filename=inspect.getfile(func),
                #                        lineno=fileno)

                frame = inspect.currentframe().f_back

                warnings.warn_explicit(message,
                                       category=DeprecatedWarning,
                                       filename=inspect.getfile(frame.f_code),
                                       lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator
