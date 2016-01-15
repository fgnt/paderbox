import unittest

from nt.chainer_models.mask_estimation.mask_estimation_template import \
    MaskEstimation
from nt.utils.chime import get_chime_data_provider_for_flist


class TestMaskEstimationTemplate(unittest.TestCase):
    def setUp(self):
        self.model = MaskEstimation()

    def test_transformation_train(self):
        dp = get_chime_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_train)
        batch = dp.test_run()
        for data_name in ['X_masks', 'N_masks', 'Y_psd', 'Y']:
            self.assertIn(data_name, batch)
            self.assertEqual(batch[data_name].shape[1], 6)
            self.assertEqual(batch[data_name].shape[2], 513)

    def test_transformation_test_simu(self):
        dp = get_chime_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_test_simu)
        batch = dp.test_run()
        for data_name in ['X', 'N', 'X_masks', 'N_masks', 'Y_psd', 'Y']:
            self.assertIn(data_name, batch)
            self.assertEqual(batch[data_name].shape[1], 6)
            self.assertEqual(batch[data_name].shape[2], 513)

    def test_transformation_cv(self):
        dp = get_chime_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_cv, return_X_N=True)
        batch = dp.test_run()
        for data_name in ['X', 'N', 'X_masks', 'N_masks', 'Y_psd', 'Y']:
            self.assertIn(data_name, batch)
            self.assertEqual(batch[data_name].shape[1], 6)
            self.assertEqual(batch[data_name].shape[2], 513)

    def test_transform_features_test_no_context(self):
        dp = get_chime_data_provider_for_flist(
                'et05_simu', self.model.transform_features_test)
        batch = dp.test_run()
        for data_name in ['Y_psd', 'Y']:
            self.assertIn(data_name, batch)
            self.assertEqual(batch[data_name].shape[1], 6)
            self.assertEqual(batch[data_name].shape[2], 513)

    def test_transform_features_test_context(self):
        dp = get_chime_data_provider_for_flist(
                'et05_real', self.model.transform_features_test)
        batch = dp.test_run()
        for data_name in ['Y_psd', 'Y']:
            self.assertIn(data_name, batch)
            self.assertEqual(batch[data_name].shape[1], 6)
            self.assertEqual(batch[data_name].shape[2], 513)
        self.assertIn('_context_samples', batch)
        self.assertGreater(batch['_context_samples'], 0)
