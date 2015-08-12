import unittest
import numpy
import numpy.testing as nptest
from nt.nn.data_fetchers.chime_mask_data_fetcher import ChimeMaskDataFetcher

src = '/net/storage/2015/chime/chime_ref_data/data/json/chime.json'

class ChimeMaskDataFetcherTest(unittest.TestCase):

    def setUp(self):
        self.fetcher = ChimeMaskDataFetcher('chime',
                            src,
                            'train/A_database/flists/wav/channels_6/tr05_simu',
                            ('X', 'masks_x', 'masks_n'),
                            X_channels=('X/CH1',),
                            N_channels=('N/CH1',))

    def test_data_type(self):
        data = self.fetcher.get_data_for_indices((0,))
        for d in data:
            self.assertTrue(d.flags.c_contiguous)
