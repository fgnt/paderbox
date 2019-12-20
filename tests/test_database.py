import unittest

import lazy_dataset
from lazy_dataset.database import DictDatabase


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.json = dict(
            datasets=dict(
                train=dict(
                    a=dict(example_id='a'),
                    b=dict(example_id='b')
                ),
                test=dict(
                    c=dict(example_id='c')
                )
            ),
            meta=dict()
        )
        # self.temp_directory = Path(tempfile.mkdtemp())
        # self.json_path = self.temp_directory / 'db.json'
        # dump_json(self.json, self.json_path)
        self.db = DictDatabase(self.json)

    def test_dataset_names(self):
        self.assertListEqual(
            list(self.db.dataset_names),
            list(self.json['datasets'].keys())
        )

    def test_dataset(self):
        dataset = self.db.get_dataset('train')
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            list(self.json['datasets']['train'].keys())
        )
        _ = dataset['a']
        _ = dataset['b']
        _ = dataset[0]
        _ = dataset[1]
        _ = dataset[:1][0]

    def test_dataset_contains(self):
        dataset = self.db.get_dataset('train')
        with self.assertRaises(Exception):
            # contains should be unsupported
            'a' in dataset

    def test_map_dataset(self):
        dataset = self.db.get_dataset('train')

        def map_fn(d):
            d['example_id'] = d['example_id'].upper()
            return d

        dataset = dataset.map(map_fn)
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'A B'.split()
        )
        _ = dataset['a']
        _ = dataset[0]
        _ = dataset[:1][0]

    def test_filter_dataset(self):
        dataset = self.db.get_dataset('train')

        def filter_fn(d):
            return not d['example_id'] == 'b'

        dataset = dataset.filter(filter_fn)
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a'.split()
        )
        _ = dataset['a']
        with self.assertRaises(IndexError):
            _ = dataset['b']
        with self.assertRaises(AssertionError):
            _ = dataset[0]
        with self.assertRaises(AssertionError):
            _ = dataset[:1]

    def test_concatenate_dataset(self):
        train_dataset = self.db.get_dataset('train')
        test_dataset = self.db.get_dataset('test')
        dataset = train_dataset.concatenate(test_dataset)
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b c'.split()
        )
        self.assertEqual(
            dataset['a']['example_id'],
            'a'
        )
        self.assertEqual(
            dataset[0]['example_id'],
            'a'
        )
        _ = dataset[:1][0]

    def test_concatenate_dataset_double_keys(self):
        train_dataset = self.db.get_dataset('train')
        dataset = train_dataset.concatenate(train_dataset)
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b a b'.split()
        )
        with self.assertRaises(AssertionError):
            _ = dataset['a']
        self.assertEqual(
            dataset[0]['example_id'],
            'a'
        )
        _ = dataset[:1][0]

    def test_multiple_concatenate_dataset(self):
        train_dataset = self.db.get_dataset('train')
        dataset = train_dataset.concatenate(train_dataset)
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b a b'.split()
        )
        _ = dataset[:1][0]

    def test_zip_dataset(self):
        import numpy as np
        train_dataset = self.db.get_dataset('train')

        # Change the key order
        np.random.seed(2)
        train_dataset_2 = train_dataset.shuffle(False)

        dataset = lazy_dataset.key_zip(train_dataset, train_dataset_2)
        dataset_2 = lazy_dataset.key_zip(train_dataset_2, train_dataset)


        example_ids = [e['example_id'] for e in train_dataset]
        self.assertListEqual(
            example_ids,
            'a b'.split()
        )

        example_ids = [e['example_id'] for e in train_dataset_2]
        self.assertListEqual(
            example_ids,
            'b a'.split()  # train_dataset_2 has swapped keys
        )
        self.assertEqual(  # dataset defined order
            list(dataset),
            [({'dataset': 'train', 'example_id': 'a'},
              {'dataset': 'train', 'example_id': 'a'}),
             ({'dataset': 'train', 'example_id': 'b'},
              {'dataset': 'train', 'example_id': 'b'})]
        )
        self.assertEqual(  # train_dataset_2 defined order
            list(dataset_2),
            [({'dataset': 'train', 'example_id': 'b'},
              {'dataset': 'train', 'example_id': 'b'}),
             ({'dataset': 'train', 'example_id': 'a'},
              {'dataset': 'train', 'example_id': 'a'})]
        )

    def test_slice_dataset(self):
        base_dataset = self.db.get_dataset('train')
        base_dataset = base_dataset.concatenate(base_dataset)
        dataset = base_dataset[:4]
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b a b'.split()
        )
        dataset = base_dataset[:3]
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b a'.split()
        )
        dataset = base_dataset[:5]  # Should this work?
        example_ids = [e['example_id'] for e in dataset]
        self.assertListEqual(
            example_ids,
            'a b a b'.split()
        )
        _ = base_dataset[:2]
        _ = base_dataset[:1]
        _ = base_dataset[:0]  # Should this work?

    # def tearDown(self):
    #     shutil.rmtree(str(self.temp_directory))


class UniqueIDDatasetTest(unittest.TestCase):
    def setUp(self):
        self.d = dict(
            datasets=dict(
                train=dict(
                    a=dict(example_id='a'),
                    b=dict(example_id='b')
                ),
                test=dict(
                    a=dict(example_id='a')
                )
            ),
            meta=dict()
        )
        self.db = DictDatabase(self.d)

    def test_duplicate_id(self):
        with self.assertRaises(AssertionError):
            dataset = self.db.get_dataset('train test'.split())
            _ = dataset.keys()

    def test_duplicate_id_with_prepend_dataset_name(self):
        _ = self.db.get_dataset('train test'.split())
