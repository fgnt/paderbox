
import line_profiler
import memory_profiler
import cProfile
import time
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from inspect import isclass, isfunction

def timefunc(func):
    def profiled_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, 'took', end - start, 'time')
        return result
    return profiled_func


def do_cprofile(func_or_str='tottime'):
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


def do_graphprofile(func):
    def profiled_func(*args, **kwargs):
        with PyCallGraph(output=GraphvizOutput()):
            return func(*args, **kwargs)
    return profiled_func


def do_lineprofile(func_or_list=list()):
    if isfunction(func_or_list):
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


def do_memprofile(func_or_list=list()):
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

    @do_cprofile("ncall")
    #@do_graphprofile
    #@do_memprofile([fibonacci])
    #@timefunc
    def example_func():
        fib = 0
        for value in fibonacci(100):
            fib = value
        return fib
    print(example_func())