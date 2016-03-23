import os
import tempfile
import time
import unittest
import warnings

import numpy as np
import numpy.testing as nptest
from chainer.functions import mean_squared_error
from chainer.link import Chain
from chainer.links import Linear
from chainer.optimizer import GradientClipping
from chainer.optimizers import SGD
from chainer.testing import attr

from nt.nn import DataProvider
from nt.nn import Trainer
from nt.nn.data_fetchers import ArrayDataFetcher
from nt.nn.training_status_utils import status_dict

B = 10
A = 5


class DummyNetwork(Chain):
    def __init__(self):
        super(DummyNetwork, self).__init__(
            l1=Linear(5, 3),
            l2=Linear(3, 5)
        )

    def forward(self, **kwargs):
        net_in = kwargs.pop('i')
        target = kwargs.pop('t')

        h = self.l1(net_in)
        h = self.l2(h)
        return mean_squared_error(h, target)

    def forward_train(self, a=1, b=2, **kwargs):
        return dict(
            net=self.forward(**kwargs),
            a=a,
            b=b)

    def forward_cv(self, a=1, b=2, **kwargs):
        return dict(
            net=self.forward(**kwargs),
            a=a,
            b=b)


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.input = np.random.uniform(-1, 1, (B, A)).astype(np.float32)
        self.target = self.input.copy()
        self.nn = DummyNetwork()
        x_fetcher = ArrayDataFetcher('i', self.input)
        t_fetcher = ArrayDataFetcher('t', self.target)
        x_cv_fetcher = ArrayDataFetcher('i', self.input)
        t_cv_fetcher = ArrayDataFetcher('t', self.target)
        self.tr_provider = DataProvider((x_fetcher, t_fetcher), batch_size=2)
        self.cv_provider = DataProvider((x_cv_fetcher, t_cv_fetcher),
                                        batch_size=2)
        hooks = [GradientClipping(1)]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.trainer = Trainer(self.nn,
                               forward_fcn_tr=self.nn.forward_train,
                               forward_fcn_cv=self.nn.forward_cv,
                               data_provider_tr=self.tr_provider,
                               data_provider_cv=self.cv_provider,
                               optimizer=SGD(),
                               description='unittest',
                               data_dir=self.tmpdir.name,
                               train_kwargs={'a': 2, 'b': 3},
                               cv_kwargs={'a': 4, 'b': 5},
                               epochs=100,
                               use_gpu=False,
                               loss_name_tr='net',
                               loss_name_cv='net',
                               optimizer_hooks=hooks)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init(self):
        self.assertTrue(os.path.exists(self.trainer.data_dir))

    def test_training_step(self):
        self.trainer.optimizer.setup(self.nn)
        batch = self.tr_provider.test_run()
        self.trainer._train_forward_batch(batch)
        self.trainer._reset_gradients()
        nptest.assert_equal(self.nn.l2.W.grad, np.zeros((5, 3)))
        self.trainer._backprop()
        W2_before = self.nn.l2.W.data.copy()
        self.trainer._update_parameters()
        self.assertRaises(
            AssertionError, nptest.assert_equal, W2_before,
            self.nn.l2.W.data)
        self.trainer._reset_gradients()
        nptest.assert_equal(self.nn.l2.W.grad, np.zeros((5, 3)))

    def test_tr_kwargs_called(self):
        self.trainer.optimizer.setup(self.nn)
        batch = self.tr_provider.test_run()
        self.trainer._train_forward_batch(batch)
        net_out = self.trainer.current_net_out
        self.assertEqual(net_out['a'], 2)
        self.assertEqual(net_out['b'], 3)

    def test_cv_kwargs_called(self):
        self.trainer.optimizer.setup(self.nn)
        batch = self.tr_provider.test_run()
        self.trainer._cv_forward_batch(batch)
        net_out = self.trainer.current_net_out
        self.assertEqual(net_out['a'], 4)
        self.assertEqual(net_out['b'], 5)

    def test_warn_on_wrong_kwarg_tr(self):
        with warnings.catch_warnings(record=True) as w:
            Trainer(self.nn,
                    forward_fcn_tr=self.nn.forward_train,
                    forward_fcn_cv=self.nn.forward_cv,
                    data_provider_tr=self.tr_provider,
                    data_provider_cv=self.cv_provider,
                    optimizer=SGD(),
                    description='unittest',
                    data_dir=tempfile.mkdtemp(),
                    train_kwargs={'a': 2, 'c': 3},
                    cv_kwargs={'a': 4, 'b': 5},
                    epochs=100,
                    use_gpu=False,
                    loss_name_tr='net',
                    loss_name_cv='net')
            self.assertEqual(len(w), 1)
            self.assertIn('defined in train_kwargs but not part of',
                          str(w[-1].message))

    def test_warn_on_wrong_kwarg_cv(self):
        with warnings.catch_warnings(record=True) as w:
            Trainer(self.nn,
                    forward_fcn_tr=self.nn.forward_train,
                    forward_fcn_cv=self.nn.forward_cv,
                    data_provider_tr=self.tr_provider,
                    data_provider_cv=self.cv_provider,
                    optimizer=SGD(),
                    description='unittest',
                    data_dir=tempfile.mkdtemp(),
                    train_kwargs={'a': 2, 'b': 3},
                    cv_kwargs={'a': 4, 'c': 5},
                    epochs=100,
                    use_gpu=False,
                    loss_name_tr='net',
                    loss_name_cv='net')
            self.assertEqual(len(w), 1)
            self.assertIn('defined in cv_kwargs but not part of',
                          str(w[-1].message))

    def _test_run_stop(self, use_gpu=False):
        self.trainer.run_in_thread = True
        if use_gpu:
            self.trainer.use_gpu = True
        self.trainer.start_training()
        self.assertTrue(self.trainer.is_running)
        self.assertTrue(self.trainer.is_running)
        self.trainer.stop_training()
        self.assertTrue(not self.trainer.is_running)

    def test_run_stop_cpu(self):
        self._test_run_stop()

    def test_run_stop_gpu(self):
        self._test_run_stop(True)

    def _test_request(self, use_gpu=False):
        self.trainer.run_in_thread = True
        if use_gpu:
            self.trainer.use_gpu = True
        self.trainer.start_training()
        time.sleep(2)
        responds = self.trainer.get_status()
        self.assertEqual(responds.description, 'unittest')
        for var in responds.__dict__.keys():
            if isinstance(var, float):
                self.assertGreater(var, 0)
            if isinstance(var, int):
                self.assertGreater(var, 0)
            if isinstance(var, list):
                self.assertGreater(len(var), 0)
            if isinstance(var, np.ndarray):
                self.assertTrue(np.any(var > 0))
        self.trainer.stop_training()

    def test_request_cpu(self):
        self._test_request()

    @attr.gpu
    def test_request_gpu(self):
        self._test_request(True)

    @attr.gpu
    def test_gpu(self):
        self.trainer.use_gpu = True
        self.trainer.run_in_thread = True
        self.trainer.start_training()
        time.sleep(1)
        self.assertTrue(self.trainer.is_running)
        self.trainer.stop_training()
        self.assertTrue(not self.trainer.is_running)

    def test_resume(self):
        epoch = 1
        tries = 0
        while (epoch < 2) and (tries < 10):
            self.trainer.start_training()
            time.sleep(tries + 1)
            self.trainer.stop_training()
            epoch = self.trainer.training_status.epoch
            if epoch is None:
                epoch = 1
            tries += 1
        self.assertGreater(10, tries)
        self.trainer.resume = True
        self.trainer.start_training()
        self.trainer.stop_training()
        self.assertGreaterEqual(self.trainer.training_status.epoch, epoch)

    def test_exception_on_data_dir_exists(self):
        self.trainer.start_training()
        time.sleep(1)
        self.trainer.stop_training()
        self.assertRaises(EnvironmentError, Trainer, *[self.nn],
                          **dict(forward_fcn_tr=self.nn.forward_train,
                                 forward_fcn_cv=self.nn.forward_cv,
                                 data_provider_tr=self.tr_provider,
                                 data_provider_cv=self.cv_provider,
                                 optimizer=SGD(),
                                 description='unittest',
                                 data_dir=self.trainer.data_dir))

    @attr.gpu
    def test_test_mode(self):
        self.trainer.test_run()
        self.trainer.use_gpu = True
        self.trainer.test_run()
        self.trainer.test_run()
        self.assertTrue(True)

    def _test_modes(self, use_gpu=False):
        self.trainer.run_in_thread = True
        if use_gpu:
            self.trainer.use_gpu = True
        status = self.trainer.get_status()
        self.assertEqual(status.current_mode, 'Idle')
        self.trainer.start_training()
        status = self.trainer.get_status()
        self.assertTrue(status.current_mode == 'Train' or 'Cross-validation')
        self.trainer.stop_training()
        status = self.trainer.get_status()
        self.assertEqual(status.current_mode, 'Stopped')

    def test_modes_cpu(self):
        self._test_modes()

    @attr.gpu
    def test_modes_gpu(self):
        self._test_modes(True)

    def test_json_export(self):
        self.trainer.test_run()
        for part in status_dict:
            self.assertTrue(
                os.path.exists(self.trainer.data_dir + '/' + part + '.json'))
