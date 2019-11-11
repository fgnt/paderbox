
import line_profiler
import memory_profiler
import cProfile
import time as time
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from inspect import isclass, isfunction
from functools import wraps


def timefunc(func):
    """
    decorator to measure the execution time of the decorated function.
    """
    def profiled_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, 'took', end - start, 'time')
        return result
    return profiled_func


def cprun(func_or_str='tottime'):
    """
    decorator for the cProfile profiler
    Args:
        func_or_str: 

    Returns:

    """
    if isfunction(func_or_str):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func_or_str(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        return profiled_func
    elif isinstance(func_or_str, str):
        def inner(func):
            def profiled_func(*args, **kwargs):
                profile = cProfile.Profile()
                try:
                    profile.enable()
                    result = func(*args, **kwargs)
                    profile.disable()
                    return result
                finally:
                    profile.print_stats(sort=func_or_str)
            return profiled_func
        return inner


def gprun(func):
    """
    graph profiling, visualizes the decorated function with a call graph
    :param func:
    :return:
    """
    def profiled_func(*args, **kwargs):
        with PyCallGraph(output=GraphvizOutput()):
            return func(*args, **kwargs)
    return profiled_func


def lprun(func_or_list=list()):
    """
    line profiling, enabling the user to profile a function not decorated, itself.
    Useful to observe the behaviour of functions without calling them directly.

    Args:
        func_or_list: list of functions to observe in the decorated function

    Returns: line-by-line analysis of the profiled functions

    """
    if isfunction(func_or_list):
        @wraps(func_or_list)
        def profiled_func(*args, **kwargs):
            profiler = line_profiler.LineProfiler()
            try:
                profiler.add_function(func_or_list)
                profiler.enable_by_count()
                return func_or_list(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    elif isinstance(func_or_list, list):
        def inner(func):
            @wraps(func)
            def profiled_func(*args, **kwargs):
                profiler = line_profiler.LineProfiler()
                try:
                    if not func_or_list:
                        func_or_list.append(func_or_list)

                    for module in func_or_list:
                        if isfunction(module):
                            profiler.add_function(module)
                        elif isclass(module):
                            for k, v in module.__dict__.items():
                                if isfunction(v):
                                    profiler.add_function(v)

                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner


def mprun(func_or_list=list()):
    """
    memory profiler, line-by-line profiling of the memory usage in a
    likewise manner as lprun
    Args:
        func_or_list:

    Returns:

    """
    if isfunction(func_or_list):
        def profiled_func(*args, **kwargs):
            profiler = memory_profiler.LineProfiler()
            try:
                profiler.add_function(func_or_list)
                profiler.enable_by_count()
                return func_or_list(*args, **kwargs)
            finally:
                memory_profiler.show_results(profiler)
        return profiled_func
    elif isinstance(func_or_list, list):
        def inner(func):
            def profiled_func(*args, **kwargs):
                profiler = memory_profiler.LineProfiler()
                try:
                    if not func_or_list:
                        func_or_list.append(func)

                    for module in func_or_list:
                        if isfunction(module):
                            profiler.add_function(module)
                        elif isclass(module):
                            for k, v in module.__dict__.items():
                                if isfunction(v):
                                    profiler.add_function(v)

                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    memory_profiler.show_results(profiler)
            return profiled_func
        return inner
    else:
        raise Warning


# ========Example=========

if __name__ == "__main__":

    def fibonacci(n):
        value_old = 1
        value = 1
        for x in range(n):
            if x>1:
                temp = value
                value = value + value_old
                value_old = temp
            yield value

    #@do_cprofile("ncall")
    #@do_graphprofile
    #@do_memprofile([fibonacci])
    #@do_lineprofile([fibonacci])
    #@timefunc
    def example_func():
        fib = 0
        for value in fibonacci(100):
            fib = value
        return fib
    print(example_func())
