__author__ = 'walter'
import unittest
import tempfile
from nt.utils.ctm_transcription import read_ctm

class TestReadCTM(unittest.TestCase):
    def setUp(self):
        self.TemporaryDirectory = tempfile.TemporaryDirectory()

        # kaldi CTM
        self.kaldi_ctm_file = self.TemporaryDirectory.name + '/kaldi.ctm'
        with open(self.kaldi_ctm_file, 'w') as fid_kaldi_ctm:
            fid_kaldi_ctm.write(
                'c02c0202 1 0.30 0.20 HER\n'
                'c02c0202 1 0.50 0.29 REAL\n'
                'c02c0203 1 0.19 0.19 AND\n'
                'c02c0203 1 0.38 0.13 THERE'
            )

        self.kaldi_ctm_ref = {
            'c02c0202': [('HER', 0.30, 0.50),
                         ('REAL', 0.50, 0.79)],
            'c02c0203': [('AND', 0.19, 0.38),
                         ('THERE', 0.38, 0.51)]
        }

        # LatticeWordegmentation
        self.lmseg_ctm_file = self.TemporaryDirectory.name + '/lmseg.ctm'
        with open(self.lmseg_ctm_file, 'w') as fid_lmseg_ctm:
            fid_lmseg_ctm.write(
                'c02c0202.lat_16.000000 HHER 0.3 0.54\n'
                'c02c0202.lat_16.000000 RIY 0.54 0.71\n'
                'c02c0203.lat_16.000000 AHND 0.19 0.39\n'
                'c02c0203.lat_16.000000 DHEY 0.39 0.54'
            )

        self.lmseg_ctm_ref = {
            'c02c0202.lat_16.000000': [('HHER', 0.3, 0.54),
                                       ('RIY', 0.54, 0.71)],
            'c02c0203.lat_16.000000': [('AHND', 0.19, 0.39),
                                       ('DHEY', 0.39, 0.54)]
        }

    def test_read_kaldi_ctm(self):
        kaldi_ctm = read_ctm(self.kaldi_ctm_file,
                             pos=(0,4,2,3),
                             has_duration=True)
        self.assertDictEqual(kaldi_ctm, self.kaldi_ctm_ref)

    def test_read_lmseg_ctm(self):
        lmseg_ctm = read_ctm(self.lmseg_ctm_file)
        self.assertDictEqual(lmseg_ctm, self.lmseg_ctm_ref)

    def tearDown(self):
        self.TemporaryDirectory.cleanup()