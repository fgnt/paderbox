import unittest

import pathlib

from nt.io.data_dir import database_jsons as database_jsons_dir
from nt.database.keys import *
from nt.database.iterator import recursive_transform, ExamplesIterator, \
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
            self.missing_wav.append(f'{wav_path}')
            self._wav_complete = False
            return False

    def test_audio_files_exist(self):
        """
        Tests if all audio files mentioned in the datasets are available.
        """
        successful = True
        self.missing_wav = list()
        for dataset_key, dataset_examples in self.json[DATASETS].items():
            self._wav_complete = True
            for example_key, example in dataset_examples.items():
                recursive_transform(self._check_audio_file_exists,
                                    example[AUDIO_PATH])

            if not self._wav_complete:
                successful = False

        self.assertTrue(successful, 'Some *.wav files referenced in the '
                                    'database json are not available:\n'
                                    f'{self.missing_wav}'
                        )

    def test_structure(self):
        self.assertIn(DATASETS, self.json)

    def assert_in_example(self, keys):
        dataset = list(self.json[DATASETS].values())[0]
        example = list(dataset.values())[0]

        for key in keys:
            self.assertIn(key, example,
                          f'The key "{key}" should be present in examples')


    def assert_in_datasets(self, datasets):
        """

        :param datasets: list of keys
        :return:
        """
        assert isinstance(datasets, list), "Datasets is not a list!"

        self.assertSetEqual(set(datasets), set(self.json[DATASETS].keys()),
                            msg=f'"{datasets}" should be in DATASETS')

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


class DatabaseClassTest(unittest.TestCase):

    def check_data_available(self, database, expected_example_keys,
                             dataset_names=None):
        if not dataset_names:
            dataset_names = database.dataset_names
        iterator = database.get_iterator_by_names(dataset_names)
        for example in iterator:
            self.assertEqual(sorted(example.keys()),
                             sorted(expected_example_keys)
                             )
