import unittest
from nt.database.reader import JsonDatabase
from pathlib import Path
import shutil
import tempfile
from nt.io import dump_json


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.json = dict(
            datasets=dict(
                train='a b'.split(),
                test='c'.split()
            ),
            examples=dict(
                a=dict(example_id='a'),
                b=dict(example_id='b'),
                c=dict(example_id='c')
            ),
            meta=dict()
        )
        self.temp_directory = Path(tempfile.mkdtemp())
        self.json_path = self.temp_directory / 'db.json'
        dump_json(self.json, self.json_path)
        self.db = JsonDatabase(self.json_path)

    def test_dataset_names(self):
        self.assertListEqual(
            sorted(self.db.dataset_names),
            sorted(list(self.json['datasets'].keys()))
        )

    def test_iterator(self):
        iterator = self.db.get_iterator_by_names('train')
        example_ids = [e['example_id'] for e in iterator]
        self.assertListEqual(
            sorted(example_ids),
            sorted(self.json['datasets']['train'])
        )

    def test_map_iterator(self):
        iterator = self.db.get_iterator_by_names('train')

        def map_fn(d):
            d['example_id'] = d['example_id'].upper()
            return d

        iterator = iterator.map(map_fn)
        example_ids = [e['example_id'] for e in iterator]
        self.assertListEqual(
            sorted(example_ids),
            'A B'.split()
        )

    def test_filter_iterator(self):
        iterator = self.db.get_iterator_by_names('train')

        def filter_fn(d):
            return not d['example_id'] == 'b'

        iterator = iterator.filter(filter_fn)
        example_ids = [e['example_id'] for e in iterator]
        self.assertListEqual(
            sorted(example_ids),
            'a'.split()
        )

    def test_concatenate_iterator(self):
        train_iterator = self.db.get_iterator_by_names('train')
        test_iterator = self.db.get_iterator_by_names('test')
        iterator = train_iterator.concatenate(test_iterator)
        example_ids = [e['example_id'] for e in iterator]
        self.assertListEqual(
            sorted(example_ids),
            'a b c'.split()
        )

    def test_multiple_concatenate_iterator(self):
        train_iterator = self.db.get_iterator_by_names('train')
        iterator = train_iterator.concatenate(train_iterator)
        example_ids = [e['example_id'] for e in iterator]
        self.assertListEqual(
            sorted(example_ids),
            sorted('a b a b'.split())
        )

    def tearDown(self):
        shutil.rmtree(str(self.temp_directory))
