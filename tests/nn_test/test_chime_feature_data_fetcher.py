import unittest
import numpy
import numpy.testing as nptest
from nt.nn.data_fetchers.chime_feature_data_fetcher import ChimeFeatureDataFetcher

src = '/net/storage/2015/chime/chime_ref_data/data/json/chime.json'

class ChimeFeatureDataFetcherTest(unittest.TestCase):

    def setUp(self):
        self.fetcher = ChimeFeatureDataFetcher('chime',
                            src,
                            'train/A_database/flists/wav/channels_6/tr05_simu',
                            feature_channels=['observed/CH1'])

    def test_data_type(self):
        data = self.fetcher.get_data_for_indices((0,))
        self.assertTrue(data.flags.c_contiguous)

    def test_data_shape(self):
        data = self.fetcher.get_data_for_indices((0,))
        self.assertEqual(data.shape[1], 1)
        self.assertEqual(data.shape[2], 1)
        self.fetcher.feature_type = 'fbank'
        data = self.fetcher.get_data_for_indices((0,))
        self.assertEqual(data.shape[1], 1)
        self.assertEqual(data.shape[2], 23)
        self.fetcher.feature_type = 'mfcc'
        data = self.fetcher.get_data_for_indices((0,))
        self.assertEqual(data.shape[1], 1)
        self.assertEqual(data.shape[2], 13)