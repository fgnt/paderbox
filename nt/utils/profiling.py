
from line_profiler import LineProfiler
import types

def do_profile(modules_under_test=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()

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
def get_number():
    for x in range(1000):
        yield x

@do_profile(modules_under_test=[get_number])
def some_calculation():
    for x in get_number():
        i = 2*x
    return 'some result!'

if __name__ == "__main__":
    result = some_calculation()