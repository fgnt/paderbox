import unittest
from nt.io.kaldi import ArkWriter, import_feature_data, \
    make_mfcc_features, import_feat_scp, make_fbank_features
import numpy as np
import tempfile

from nt.io.data_dir import testing as data_dir

WAV_SCP = data_dir.join('kaldi_io/wav.scp')

class KaldiIOTest(unittest.TestCase):

    def setUp(self):
        self.arr1 = np.random.normal(0, 1, (100, 30))
        self.arr2 = np.random.normal(0, 1, (200, 30))

    def test_write_read(self):
        with tempfile.NamedTemporaryFile('w') as f:
            with ArkWriter(f.name) as writer:
                writer.write_array('arr1', self.arr1)
                writer.write_array('arr2', self.arr2)
            data = import_feature_data(f.name)
            self.assertIn('arr1', data)
            self.assertIn('arr2', data)
            np.testing.assert_almost_equal(data['arr1'], self.arr1, decimal=5)
            np.testing.assert_almost_equal(data['arr2'], self.arr2, decimal=5)

    def test_make_mfccs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_mfcc_features(WAV_SCP, tmp_dir, num_mel_bins=23, num_ceps=13)
            data = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                self.assertEqual(d.shape[1], 39)

    def test_make_mfccs_no_delta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_mfcc_features(WAV_SCP, tmp_dir, num_mel_bins=23, num_ceps=13,
                               add_deltas=False)
            data = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                self.assertEqual(d.shape[1], 13)

    def test_make_fbank(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_fbank_features(WAV_SCP, tmp_dir, num_mel_bins=23)
            data = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                self.assertEqual(d.shape[1], 3*23)

    def test_make_fbank_no_delta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_fbank_features(WAV_SCP, tmp_dir, num_mel_bins=23,
                               add_deltas=False)
            data = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                self.assertEqual(d.shape[1], 23)