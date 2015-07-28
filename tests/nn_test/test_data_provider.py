from nt.nn.data_provider import DataProvider
from nt.nn.data_fetchers.data_fetcher import DataFetcher
import numpy
import numpy.testing
import unittest

class IdentityFetcher(DataFetcher):

    def __init__(self, name, len=10):
        DataFetcher.__init__(self, name)
        self.len = len

    def __len__(self):
        return self.len

    def get_data_for_indices(self, idxs):
        return numpy.asarray(list(idxs))


class DataProviderFetcher(unittest.TestCase):

    def setUp(self):
        self.fetcher_1 = IdentityFetcher('X')
        self.fetcher_2 = IdentityFetcher('Y')
        self.dp = DataProvider((self.fetcher_1, self.fetcher_2),
                               batch_size=2,
                               max_queue_size=5,
                               shuffle_data=False)

    def test_setup(self):
        self.assertEqual(len(self.dp), 5)

    def test_iteration(self):
        for idx, batch_data in enumerate(self.dp):
            self.assertTrue('X' in batch_data)
            self.assertTrue('Y' in batch_data)
            numpy.testing.assert_equal(batch_data['X'],
                                       numpy.asarray([idx*2, idx*2+1]))
            numpy.testing.assert_equal(batch_data['Y'],
                                       numpy.asarray([idx*2, idx*2+1]))

    def test_reset(self):
        self.test_iteration()
        self.test_iteration()

    def test_random(self):
        self.dp.shuffle_data = True
        self.assertRaises(AssertionError, self.test_iteration)

    def test_different_length(self):
        self.fetcher_2.len = 11
        def create_dp():
            self.dp = DataProvider((self.fetcher_1, self.fetcher_2),
                               batch_size=2,
                               max_queue_size=5,
                               shuffle_data=False)
        self.assertRaises(EnvironmentError, create_dp)