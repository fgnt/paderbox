import queue
import concurrent.futures
from pathos.multiprocessing import ProcessPool as PathosPool
from contextlib import contextmanager
import dill


def _dill_mp_helper(payload):
    fun, args, kwargs = dill.loads(payload)
    return fun(*args, **kwargs)


def lazy_parallel_map(
        function,
        generator,
        *,
        args=[],
        kwargs={},
        backend="mp",
        buffer_size=5,
        max_workers=2
):
    """
    This is a parallel map where the function is parallel executed and the
    output is bufferd. Note the generator is executed serial.

    A serial version of this function is:
        for ele in generator:
            yield function(ele, *args, **kwargs)
    The function is executed in parallel (not the generator) and the output of
    the function is bufferd.

    Note:
     - The overhead to copy the data from and to the workers can destroy the
       gain from multiprosess ('mp', 'dill_mp').
       Only the threaded backend ('t') has no copy overhead.
     - When the function spends a high amount of time in numpy and/or I/O the
       threaded backend ('t') is recommended. The reason for numpy is, that it
       usually releases the GIL.

    """
    q = queue.Queue()

    if backend is not False:
        assert buffer_size >= max_workers
    assert buffer_size > 0

    if backend == "mp":
        # http://stackoverflow.com/a/21345423
        PoolExecutor = PathosPool

        def submit(ex, func, *args, **kwargs):
            return ex.apipe(func, *args, **kwargs)

        def result(job):
            return job.get()

    elif backend == "dill_mp":
        # https://stackoverflow.com/a/24673524
        PoolExecutor = concurrent.futures.ProcessPoolExecutor

        def submit(ex, func, *args, **kwargs):
            payload = dill.dumps((func, args, kwargs))
            return ex.submit(_dill_mp_helper, payload)

        def result(job):
            return job.result()

    elif backend in [
            "t",
            "thread"
            "concurrent_mp"
    ]:
        if backend in ['t', 'thread']:
            PoolExecutor = concurrent.futures.ThreadPoolExecutor
        elif backend == "concurrent_mp":
            # does not allow to pickle arbitary functions
            PoolExecutor = concurrent.futures.ProcessPoolExecutor
        else:
            raise ValueError(backend)

        def submit(ex, func, *args, **kwargs):
            return ex.submit(func, *args, **kwargs)

        def result(job):
            return job.result()
    # elif backend is False:
    #
    #     @contextmanager
    #     def PoolExecutor(max_workers):
    #         yield None
    #
    #     def submit(ex, func, *args, **kwargs):
    #         return func(*args, **kwargs)
    #
    #     def result(job):
    #         return job
    else:
        raise ValueError(backend)

    with PoolExecutor(max_workers) as executor:
        # First fill the buffer
        # If buffer full, take one element and push one new inside
        for ele in generator:
            # print(q.qsize(), 'q.qsize()', buffer_size)
            if q.qsize() >= buffer_size:
                # print('y1')
                yield result(q.get())
            q.put(submit(executor, function, ele, *args, **kwargs))
        while not q.empty():
            yield result(q.get())


if __name__ == '__main__':
    import time

    def identity(x):
        return x


    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for i in lazy_parallel_map(identity, range(10)):
        print(i, end=' ')
    print()

    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for backend in [
        't',
        'mp',
        'concurrent_mp',
        # False,
    ]:

        for i in lazy_parallel_map(identity, range(10), backend=backend):
            print(i, end=' ')
        print()


    def task(i):
        time.sleep(0.1)
        return i


    print(f'Expect: ' + ' '.join(map(str, range(10))))
    print('Got:    ', end='')
    for i in lazy_parallel_map(lambda x: x, range(10), backend='dill_mp'):
        print(i, end=' ')
    print()

    from nt.utils.timer import Timer
    t = Timer(verbose=True)

    print(f'Serial time: ')
    print('Got:    ', end='')
    with t:
        for i in lazy_parallel_map(task, range(10), backend='dill_mp',
                                   buffer_size=5, max_workers=2):
            print(i)
            # print(i, end=' ')

    # Does not work
    # for i in lazy_parallel_map(lambda x: x, range(10), backend='concurrent_mp'):
    #     print(i, end=' ')
    # print()
