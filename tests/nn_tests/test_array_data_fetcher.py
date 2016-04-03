from nt.nn.data_provider import DataProvider
from nt.nn.data_fetchers import ArrayDataFetcher
import numpy
import numpy.testing
import unittest

class ArrayDataFetcherTest(unittest.TestCase):

    def setUp(self):
        self.data = numpy.repeat(numpy.arange(5)[None, :], 10, axis=0)
        self.data_2 = numpy.repeat(numpy.arange(5,0,-1)[None, :], 10, axis=0)
        self.fetcher_1 = ArrayDataFetcher('X', self.data)
        self.fetcher_2 = ArrayDataFetcher('Y', self.data_2)
        self.dp = DataProvider((self.fetcher_1, self.fetcher_2),
                               batch_size=2,
                               max_queue_size=5,
                               shuffle_data=False)
        ## With Context
        self.data_3 = numpy.repeat(numpy.arange(5)[None, :], 10, axis=0)
        self.data_4 = numpy.repeat(numpy.arange(1, 0, -1)[None, :], 10, axis=0)
        self.bins = numpy.array([0, self.data_3.shape[0]])
        self.fetcher_3 = ArrayDataFetcher('Xc', self.data_3, self.bins, with_context=True)
        self.fetcher_4 = ArrayDataFetcher('Yc', self.data_4)
        self.dp_2 = DataProvider((self.fetcher_3, self.fetcher_4),
                                 batch_size=2,
                                 max_queue_size=5,
                                 shuffle_data=False)


    def test_setup(self):
        self.assertEqual(len(self.dp), 5)
        self.assertEqual(len(self.dp_2), 5)

    def test_iteration(self):
        for idx, batch_data in enumerate(self.dp.iterate()):
            self.assertTrue('X' in batch_data)
            self.assertTrue('Y' in batch_data)
            numpy.testing.assert_equal(batch_data['X'],
                                       self.data[(idx*2, idx*2+1), ...])
            numpy.testing.assert_equal(batch_data['Y'],
                                       self.data_2[(idx*2, idx*2+1), ...])

    def test_reset(self):
        self.test_iteration()
        self.test_iteration()

    def test_data(self):
        batch = self.dp.test_run()
        for data in batch.values():
            if isinstance(data, numpy.ndarray):
                self.assertTrue(data.flags.c_contiguous)

    def test_get_shape(self):
        s = self.fetcher_1.get_data_shape()
        self.assertEqual(s, {'X': (1, 5)})

    def test_data_withContext(self):
        batch = self.dp_2.test_run()
        for data in batch.values():
            if isinstance(data, numpy.ndarray):
                self.assertTrue(data.flags.c_contiguous)

    def test_get_shape_withContext(self):
        s = self.fetcher_3.get_data_shape()
        self.assertEqual(s, {'Xc': (1, 55)})
