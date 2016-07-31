#!/usr/bin/env python

import sys
import numpy as np
from nt.nn import DataProvider
from nt.nn.data_fetchers import ArrayDataFetcher
import inspect
import multiprocessing as mp

if __name__ == '__main__':
    """
    ToDo: make this to an integration test

    This Test would break, if the DataProvider does not close all
    multiprosessing queues.

    This is not a regular test, because it needs many minutes to execute.

    The buggy version of the data_provider broke at i between 20 and 30.
    """

    # l = []
    # for i in range(10000):
    #     l += [mp.Queue()]
    #     print(i, end=' ')

    def __line__():
        f = sys._getframe().f_back
        return f.f_lineno + f.f_code.co_firstlineno

    print(__line__())

    print('Start')

    B = 1
    A = 1
    inputs = np.random.uniform(-1, 1, (B, A)).astype(np.float32)
    targets = inputs.copy()
    t_cv_fetcher = [ArrayDataFetcher('t' + str(i), targets.copy()) for i in
                    range(2)]
    cv_provider = DataProvider(t_cv_fetcher, batch_size=1)

    for i in range(10000):
        for _ in cv_provider.iterate(fork_fetchers=True):
            # if i % 10 == 0:
            print(i, end=' ')
