import unittest
from nt.io.kaldi import (ArkWriter, import_feature_data,
    make_mfcc_features, import_feat_scp, make_fbank_features,
    compute_mean_and_var_stats, apply_mean_and_var_stats, read_scp_file)
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

            def _test():
                self.assertIn('arr1', data)
                self.assertIn('arr2', data)
                np.testing.assert_almost_equal(data['arr1'], self.arr1, decimal=5)
                np.testing.assert_almost_equal(data['arr2'], self.arr2, decimal=5)
            _test()

            data = import_feat_scp(f.name + '.scp')
            _test()

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

    def test_make_fbank_cmvn(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_fbank_features(WAV_SCP, tmp_dir, num_mel_bins=23)
            compute_mean_and_var_stats(tmp_dir + '/feats.scp', tmp_dir)
            apply_mean_and_var_stats(tmp_dir + '/feats.scp', tmp_dir + '/cmvn.ark')
            data = import_feat_scp(tmp_dir + '/normalized.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                np.testing.assert_almost_equal(np.mean(d, axis=0), 3*23*[0.],
                                               decimal=5)
                self.assertEqual(d.shape[1], 3 * 23)
            apply_mean_and_var_stats(tmp_dir + '/feats.scp',
                                     tmp_dir + '/cmvn.ark',
                                     norm_var=True)
            data = import_feat_scp(tmp_dir + '/normalized.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                np.testing.assert_almost_equal(np.mean(d, axis=0),
                                               3 * 23 * [0.],
                                               decimal=5)
                np.testing.assert_almost_equal(np.var(d, axis=0), 3 * 23 * [1.],
                                               decimal=5)
                self.assertEqual(d.shape[1], 3 * 23)

    def test_make_fbank_cmvn_utt2spk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_fbank_features(WAV_SCP, tmp_dir, num_mel_bins=23)
            scp = read_scp_file(WAV_SCP)
            speaker = ['A', 'B']
            utt2spk = {utt: speaker[idx%2] for idx, utt in enumerate(scp)}

            compute_mean_and_var_stats(tmp_dir + '/feats.scp', tmp_dir,
                                       utt2spk=utt2spk)
            apply_mean_and_var_stats(tmp_dir + '/feats.scp',
                                     tmp_dir + '/cmvn.ark',
                                     utt2spk=utt2spk)
            data = import_feat_scp(tmp_dir + '/normalized.scp')
            data_org = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d, d_org in zip(data.values(), data_org.values()):
                self.assertLess(np.sum(np.abs(np.mean(d, axis=0))),
                                np.sum(np.abs(np.mean(d_org, axis=0))))
                self.assertEqual(d.shape[1], 3 * 23)

    def test_make_fbank_no_delta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            make_fbank_features(WAV_SCP, tmp_dir, num_mel_bins=23,
                               add_deltas=False)
            data = import_feat_scp(tmp_dir + '/feats.scp')
            self.assertEqual(len(data), 17)
            for d in data.values():
                self.assertEqual(d.shape[1], 23)