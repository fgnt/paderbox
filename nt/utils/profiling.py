
from line_profiler import LineProfiler
import types
import cProfile
import time


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'time')
        return result
    return f_timer

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
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

    def get_number():
        for x in range(1000):
            yield x

    @do_lineprofile(modules_under_test=[get_number])
    @do_cprofile
    def some_calculation():
        for x in get_number():
            i = 2*x
        return 'some result!'
    result = some_calculation()