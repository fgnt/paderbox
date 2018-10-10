import unittest
from nt.kaldi.io import read_keyed_text_file, write_keyed_text_file
import tempfile
from pathlib import Path


class TestReadWriteKeyedTextFile(unittest.TestCase):
    @staticmethod
    def check(data_dict, as_list: bool):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / 'text'
            write_keyed_text_file(temp_file, data_dict)
            result = read_keyed_text_file(temp_file, to_list=as_list)
        assert isinstance(result, dict), type(result)
        assert data_dict.keys() == result.keys(), (data_dict, result)
        for k in data_dict.keys():
            assert data_dict[k] == result[k], (k, data_dict, result)

    def test_with_value_as_list_of_strings(self):
        data_dict = dict(a=['a', '1'], b=['b', '2'], c=['c', '3'])
        self.check(data_dict, as_list=True)

    def test_with_value_as_single_string(self):
        data_dict = dict(a='a 1', b='c 2', c='c 3')
        self.check(data_dict, as_list=False)
