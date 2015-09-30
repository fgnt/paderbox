
from line_profiler import LineProfiler
import types
import cProfile
import time
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def timefunc(func):
    def profiled_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, 'took', end - start, 'time')
        return result
    return profiled_func


def do_cprofile(sort_type='tottime'):
    def inner(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats(sort=sort_type)
        return profiled_func
    return inner

def do_graphprofile(func):
    def profiled_func(*args, **kwargs):
        with PyCallGraph(output=GraphvizOutput()):
            return func(*args, **kwargs)
    return profiled_func


def do_lineprofile(modules_under_test=list()):
    def inner(func):
        def profiled_func(*args, **kwargs):
            profiler = LineProfiler()
            try:

                if not modules_under_test:
                    modules_under_test.append(func)

                for m in modules_under_test:
                    if type(m) is types.FunctionType:
                        profiler.add_function(m)
                    else:
                        profiler.add_module(m)

                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                print("\n" + 10*"=" + " Profiling " + 10*"=")
                profiler.print_stats()
        return profiled_func
    return inner


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
    @do_graphprofile
    #@do_lineprofile([fibonacci])
    #@timefunc
    def example_func():
        fib = 0
        for value in fibonacci(100):
            fib = value
        return fib
    print(example_func())