from nt.nn.data_provider import DataProvider
from nt.nn.data_fetchers.data_fetcher import DataFetcher
import numpy
import numpy.testing
import unittest
import pandas

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

    def test_data_shapes(self):
        s = self.dp.get_data_shapes()
        self.assertEqual(s['X'], (1,))
        self.assertEqual(s['Y'], (1,))

    def test_data_types(self):
        s = self.dp.get_data_types()
        self.assertEqual(s['X'], numpy.int64)
        self.assertEqual(s['Y'], numpy.int64)

    def test_data_info(self):
        df = self.dp.data_info

        reference_df = pandas.DataFrame([
            {'Fetcher Name': 'X', 'Output': 'X', 'Shape': (1,), 'Type': 'int64', 'C Contiguous': True},
            {'Fetcher Name': 'Y', 'Output': 'Y', 'Shape': (1,), 'Type': 'int64', 'C Contiguous': True}
        ])

        # sorted_df = df[sorted([col for col in df])]
        # sorted_reference_df = reference_df[sorted([col for col in reference_df])]

        # print(sorted_df)
        # print(sorted_reference_df)

        # print(sorted_df.equals(sorted_reference_df))

        # self.assertTrue(sorted_df.equals(sorted_reference_df))

        # assert list(df['Fetcher Name']) == ['X', 'Y'], 'Fetcher Names mismatch.'

        assert False

    # TODO: Missing tests for get_data_types, get_data_shapes, print_data_info
