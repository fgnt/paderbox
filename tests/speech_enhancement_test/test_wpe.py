import unittest
from nt.utils.random import uniform, hermitian
import nt.testing as tc
from nt.utils.math_ops import cos_similarity
import numpy as np




class TestWPEWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.settings_file_path='/net/home/wilhelmk/PythonToolbox/nt/speech_enhancement/utils/'
        self.input_file_paths={'/net/storage/python_unittest_data/speech_enhancement/data/sample_ch1.wav':1,}
        self.output_dir_path='/net/storage/python_unittest_data/speech_enhancement/data/'


    def test_dereverb_1ch(self):
        pass

