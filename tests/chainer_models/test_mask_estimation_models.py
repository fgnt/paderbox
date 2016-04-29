import unittest
from tempfile import TemporaryDirectory

from chainer.optimizers import Adam
from nt.database.chime import get_data_provider_for_flist

from nt.chainer_models.mask_estimation.chime_paper import ChimePaperModel
from nt.chainer_models.mask_estimation.cnn_models import BasicCNNChannelModel
from nt.chainer_models.mask_estimation.cnn_models import BasicCNNModel
from nt.nn import Trainer


class TestChimePaperModel(unittest.TestCase):
    def setUp(self):
        self.model = ChimePaperModel()
        self.tmp_dir = TemporaryDirectory()
        self.dp_train = get_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_train,
                load_images=True
        )
        self.dp_cv = get_data_provider_for_flist(
                'dt05_simu', self.model.transform_features_cv,
                load_images=True
        )
        self.trainer = Trainer(
                network=self.model,
                data_provider_tr=self.dp_train,
                data_provider_cv=self.dp_cv,
                optimizer=Adam(),
                description='ChimePaperModel',
                data_dir=self.tmp_dir.name,
                forward_fcn_tr=self.model.train,
                forward_fcn_cv=self.model.train,
                epochs=200,
                use_gpu=False,
                train_kwargs={},
                cv_kwargs={},
                run_in_thread=True,
                retain_gradients=True,
                patience=15,
        )

    def test_test_run(self):
        self.trainer.test_run()
        self.assertGreater(self.trainer.current_net_out['loss'].num, 0.)
        self.assertGreater(2., self.trainer.current_net_out['loss'].num)
        for data_name in ['X_masks', 'N_masks', 'X_mask_hat', 'N_mask_hat']:
            self.assertIn(data_name, self.trainer.current_net_out)
            self.assertEqual(
                    self.trainer.current_net_out[data_name].num.shape[2], 513)

    def test_calc_masks(self):
        batch = self.dp_train.test_run()
        N_masks, X_masks = self.model.calc_masks(batch)
        self.assertEqual(N_masks.shape[1:], (6, 513))
        self.assertEqual(X_masks.shape[1:], (6, 513))

    def test_calc_mask(self):
        batch = self.dp_train.test_run()
        N_masks, X_masks = self.model.calc_mask(batch)
        self.assertEqual(N_masks.shape[1:], (513,))
        self.assertEqual(X_masks.shape[1:], (513,))

    def tearDown(self):
        self.tmp_dir.cleanup()

class TestBasicCNNChannelModel(TestChimePaperModel):

    def setUp(self):
        self.model = BasicCNNChannelModel()
        self.tmp_dir = TemporaryDirectory()
        self.dp_train = get_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_train,
                load_images=True
        )
        self.dp_cv = get_data_provider_for_flist(
                'dt05_simu', self.model.transform_features_cv,
                load_images=True
        )
        self.trainer = Trainer(
                network=self.model,
                data_provider_tr=self.dp_train,
                data_provider_cv=self.dp_cv,
                optimizer=Adam(),
                description='ChimePaperModel',
                data_dir=self.tmp_dir.name,
                forward_fcn_tr=self.model.train,
                forward_fcn_cv=self.model.train,
                epochs=200,
                use_gpu=False,
                train_kwargs={},
                cv_kwargs={},
                run_in_thread=True,
                retain_gradients=True,
                patience=15,
        )

    def test_calc_masks(self):
        batch = self.dp_train.test_run()
        N_masks, X_masks = self.model.calc_masks(batch)
        self.assertEqual(N_masks.shape[1:], (1, 513))
        self.assertEqual(X_masks.shape[1:], (1, 513))

class TestBasicCNNModel(TestChimePaperModel):

    def setUp(self):
        self.model = BasicCNNModel()
        self.tmp_dir = TemporaryDirectory()
        self.dp_train = get_data_provider_for_flist(
                'tr05_simu', self.model.transform_features_train,
                load_images=True
        )
        self.dp_cv = get_data_provider_for_flist(
                'dt05_simu', self.model.transform_features_cv,
                load_images=True
        )
        self.trainer = Trainer(
                network=self.model,
                data_provider_tr=self.dp_train,
                data_provider_cv=self.dp_cv,
                optimizer=Adam(),
                description='ChimePaperModel',
                data_dir=self.tmp_dir.name,
                forward_fcn_tr=self.model.train,
                forward_fcn_cv=self.model.train,
                epochs=200,
                use_gpu=False,
                train_kwargs={},
                cv_kwargs={},
                run_in_thread=True,
                retain_gradients=True,
                patience=15,
        )
