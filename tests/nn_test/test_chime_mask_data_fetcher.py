import unittest
import numpy
import numpy.testing as nptest
from nt.nn.data_fetchers.chime_mask_data_fetcher import ChimeMaskDataFetcher
import time

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

    def test_tmp_caching(self):
        fetcher = ChimeMaskDataFetcher('chime',
                            src,
                            'train/A_database/flists/wav/channels_6/tr05_simu',
                            ('X', 'masks_x', 'masks_n'),
                            X_channels=('X/CH1',),
                            N_channels=('N/CH1',),
                            cache_to_tmp=True)
        data = fetcher.get_data_for_indices((0,))
        self.assertGreater(len(fetcher.tmp_file_dict), 0)
        tmp_file_name = list(fetcher.tmp_file_dict.values())[0]
        data_load = fetcher.get_data_for_indices((0,))
        for d, d_l in zip(data, data_load):
            nptest.assert_equal(d, d_l)
        fetcher.clear_tmp_files()
        self.assertRaises(FileNotFoundError, open, tmp_file_name)
