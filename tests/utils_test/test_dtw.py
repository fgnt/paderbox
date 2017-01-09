import unittest
from nt.utils.dtw import dtw
from nt.transform import fbank
from nt.io.audioread import read_raw
from nt.math.vector import cos_distance
from numpy import array
from numpy.testing import assert_equal
from nt.io.data_dir import tidigits


class TestDTW(unittest.TestCase):
    def test_dtw(self):
        file1 = tidigits / 'tidigits_16kHz_LE' / 'train' / 'man' / 'ae' / 'oa.raw'
        file2 = tidigits / 'tidigits_16kHz_LE' / 'train' / 'man' / 'ae' / 'ob.raw'
        audio1 = read_raw(file1)
        audio2 = read_raw(file2)
        fb1 = fbank(audio1)
        fb2 = fbank(audio2)
        dist_min, dist_mat, dist_acc, path = dtw(fb1, fb2, cos_distance)

        res_path = \
            (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 13,
                    14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27,
                    28, 29, 30, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                    56, 57, 58, 58, 59, 60, 61, 61, 62, 63, 64, 65, 65, 66, 67,
                    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 78, 78, 78, 79,
                    79, 79, 80, 80, 80, 80, 80, 81, 82, 83, 84, 85, 86, 86, 87,
                    88, 89]),
             array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,
                     6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 32,
                    33, 33, 34, 35, 36, 37, 38, 39, 39, 40, 40, 41, 42, 43, 43,
                    44, 45, 46, 47, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57,
                    57, 57, 57, 57, 57, 57, 57, 58, 58, 59, 60, 61, 62, 63, 64,
                    65, 66, 67, 68, 69, 70, 71, 72, 72, 72, 73, 74, 75, 76, 77,
                    78, 79]))

        assert_equal(path[0], res_path[0])
        assert_equal(path[1], res_path[1])
        self.assertAlmostEqual(dist_min, 4.7201276772539877)
