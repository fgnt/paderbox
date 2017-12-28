import unittest

import pathlib
from natsort import natsorted, ns

from nt.io.data_dir import database_jsons as database_jsons_dir
from nt.database.keys import *
from nt.database.reader import recursive_transform, ExamplesIterator, \
    AudioReader

ROOT = database_jsons_dir


class DatabaseTest(unittest.TestCase):

    @property
    def json(self):
        return self._json

    @json.setter
    def json(self, value):
        self._json = value

    def _check_audio_file_exists(self, wav_path):
        wav_path = pathlib.Path(wav_path)
        if wav_path.exists() and wav_path.is_file():
            return True
        else:
            print(f'  {wav_path}')
            self._wav_complete = False
            return False

    def test_audio_files_exist(self):
        """
        Tests if all audio files mentioned in the datasets are available.
        """
        successful = True
        for dataset_key, dataset_examples in self.json[DATASETS].items():
            print(f'{dataset_key}:')
            self._wav_complete = True
            for example_key, example in dataset_examples.items():
                recursive_transform(self._check_audio_file_exists,
                                    example[AUDIO_PATH])

            if not self._wav_complete:
                successful = False
                print('  is not complete!')
            else:
                print('  is complete!')

        assert successful, 'Some *.wav file referenced in the database '\
                           'json are not available!'

    def test_structure(self):
        self.assertIn(DATASETS, self.json)

    def assert_in_example(self, keys):
        dataset = list(self.json[DATASETS].values())[0]
        example = list(dataset.values())[0]

        for key in keys:
            self.assertIn(key, example,
                          f'The key "{key}" should be present in examples')

    #def test_natsorted(self):
    #    """
    #    Tests if datasets and examples in datasets are natsorted.
    #    """
    #    datasets = list(self.json[DATASETS].keys())
    #    natsorted_datasets = natsorted(datasets)
    #    self.assertEqual(datasets, natsorted_datasets,
    #                     "Datasets are not natsorted!")
    #    for ds in datasets:
    #        examples = list(self.json[DATASETS][ds].keys())
    #        natsorted_examples = natsorted(examples)
    #        self.assertEqual(examples, natsorted_examples,
    #                        f'Examples in dataset {ds} are not natsorted!')



    def assert_in_datasets(self, datasets):
        """

        :param datasets: list of keys
        :return:
        """
        assert isinstance(datasets, list), "Datasets is not a list!"

        for dataset in datasets:
            self.assertIn(dataset, self.json[DATASETS],
                          f'"{dataset}" should be in DATASETS')

    def test_examples(self):
        self.assert_in_example([AUDIO_PATH, ])

    def assert_total_length(self, total_length):
        _total_length = 0
        for dataset in self.json[DATASETS].values():
            _total_length += len(dataset)

        self.assertEqual(total_length, _total_length,
                         f'The database should contain exactly {total_length} '
                         f'examples in total, but contains {_total_length}')

    def test_reader(self):
        reader = AudioReader()
        for dataset_key, dataset in self.json[DATASETS].items():
            iterator = ExamplesIterator(dataset)
            example = next(iter(iterator))
            example = reader(example)

            # check if audio data was loaded
            self.assertIn(AUDIO_DATA, example)
