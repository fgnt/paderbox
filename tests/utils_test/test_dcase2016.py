import unittest
from nt.utils.dcase2016 import get_train_cv_data_provider
import nt.utils.json_utils as ju

json_path = '/net/storage/database_jsons/dcase2016.json'
json_data = ju.load_json(json_path)
flist = 'train/Complete Set/wav/mono'
transcription_list = json_data['train']['Complete Set']['annotation']['orth']
events = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn',
          'phone', 'speech']
dir_name = '/net/speechdb/DCASE2016/dcase2016_task2_train_dev/dcase2016_task2_train/'

class TestDcaseTrainAndCvDataProviders(unittest.TestCase):
    def test_no_context(self):
        dp_train, dp_cv = get_train_cv_data_provider(dir_name, 512, 160, transcription_list, events,
                                                     resampling_factor=16 / 44.1, batch_size=128,
                                                     cnn_features=False, num_fbanks=26)
        shapes = dp_train.get_data_shapes()
        assert shapes['targets'] == (1, 12)
        assert shapes['x'] == (1, 26)

    def test_with_context(self):
        left_context = 3
        right_context = 4
        dp_train, dp_cv = get_train_cv_data_provider(dir_name, 512, 160, transcription_list, events,
                                                     resampling_factor=16 / 44.1, batch_size=1500,
                                                     cnn_features=False, num_fbanks=26,
                                                     left_context=left_context, right_context=right_context)
        shapes = dp_train.get_data_shapes()
        assert shapes['targets'] == (1, 12)
        assert shapes['x'] == (1, 26 * (left_context + 1 + right_context))
