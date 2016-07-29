#!/usr/bin/env python

import numpy as np
from nt.nn import DataProvider
from nt.nn.data_fetchers import ArrayDataFetcher

if __name__ == '__main__':
    """
    ToDo: make this to an integration test

    This Test would break, if the DataProvider does not close all
    multiprosessing queues.

    This is not a regular test, because it needs many minutes to execute.

    The buggy version of the data_provider broke at i between 20 and 30.
    """

    print('Start')

    B = 1
    A = 1
    inputs = np.random.uniform(-1, 1, (B, A)).astype(np.float32)
    targets = inputs.copy()
    t_cv_fetcher = [ArrayDataFetcher('t' + str(i), targets.copy()) for i in
                    range(20)]
    cv_provider = DataProvider(t_cv_fetcher, batch_size=1)

    for i in range(1000):
        for _ in cv_provider.iterate():
            print(i, end=' ')
