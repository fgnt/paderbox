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
        return {self.name: numpy.asarray(list(idxs))}


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

    def test_error_on_same_names(self):
        def make_dp():
            return DataProvider((self.fetcher_1, self.fetcher_1),
                               batch_size=2,
                               max_queue_size=5,
                               shuffle_data=False)
        self.assertRaises(ValueError, make_dp)

    def test_iteration(self):
        for idx, batch_data in enumerate(self.dp.iterate()):
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

    def test_test_run(self):
        data = self.dp.test_run()
        numpy.testing.assert_equal(data['X'], numpy.asarray([0]))
        numpy.testing.assert_equal(data['Y'], numpy.asarray([0]))

    def test_get_indices(self):
        data = self.dp.get_data_for_indices_tuple((0, 2, 4))
        numpy.testing.assert_equal(data['X'], numpy.asarray([0, 2, 4]))
        numpy.testing.assert_equal(data['Y'], numpy.asarray([0, 2, 4]))

    def test_shutdown(self):
        self.dp._setup_data_fetchers()
        self.dp.shutdown()
        for f in self.dp.fetchers:
            self.assertTrue(len(self.dp.fetcher_processes) == 0)

    def test_data_shapes(self):
        s = self.dp.get_data_shapes()
        self.assertEqual(len(s), 2)
        self.assertTrue('X' in s)
        self.assertTrue('Y' in s)

    def test_output_list(self):
        self.assertEqual(self.dp.output_list, ['X', 'Y'])

    def test_print_data_info(self):
        self.dp.print_data_info()
        self.dp.print_data_info(format=0)

    # TODO: Missing tests for get_data_types, get_data_shapes, print_data_info