import unittest
from nt.nn import *
import numpy as np
from chainer import FunctionSet
import chainer.functions as F
from chainer.optimizers import SGD
import os
import time

B = 10
A = 5

class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork(inputs=('i', 't'), outputs=('h0', 'y', 'l'))
        self.input = np.random.uniform(-1, 1, (B, A)).astype(np.float32)
        self.target = self.input.copy()
        self.nn.layers = FunctionSet(l1=F.Linear(5, 3),
                                     l2=F.Linear(3, 5))
        x_fetcher = ArrayDataFetcher('i', self.input)
        t_fetcher = ArrayDataFetcher('t', self.target)
        x_cv_fetcher = ArrayDataFetcher('i', self.input)
        t_cv_fetcher = ArrayDataFetcher('t', self.target)
        self.tr_provider = DataProvider((x_fetcher, t_fetcher), batch_size=2)
        self.cv_provider = DataProvider((x_cv_fetcher, t_cv_fetcher),
                                        batch_size=2)
        self.optimizer = SGD(1e-5)
        self.optimizer.setup(self.nn.layers.collect_parameters())
        self.trainer = Trainer(self.nn,
                               forward_fcn=self._forward,
                               data_provider_tr=self.tr_provider,
                               data_provider_cv=self.cv_provider,
                               optimizer=self.optimizer,
                               loss_name='l',
                               description='unittest',
                               data_dir='test_tmp',
                               epochs=50,
                               logging_list=['h0'],
                               grad_clip=5,
                               use_gpu=False)

    def _forward(self, nn):
        h0 = F.sigmoid(nn.layers.l1(nn.inputs.i))
        y = nn.layers.l2(h0)
        nn.outputs.l = F.mean_squared_error(y, nn.inputs.t)
        nn.outputs.h0 = h0
        nn.outputs.y = y

    def test_init(self):
        self.assertTrue(os.path.exists(self.trainer.data_dir))

    def _test_run_stop(self, use_gpu=False):
        self.trainer.run_in_thread = True
        if use_gpu:
            self.trainer.use_gpu = True
        self.trainer.start_training()
        self.assertTrue(self.trainer.is_running)
        time.sleep(2)
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
        request = ['l', 'h0', 'y']
        time.sleep(2)
        responds = self.trainer.get_status(request)
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

    def test_request_gpu(self):
        self._test_request(True)

    def test_gpu(self):
        self.trainer.use_gpu = True
        self.trainer.run_in_thread = True
        self.trainer.start_training()
        time.sleep(1)
        self.assertTrue(self.trainer.is_running)
        self.trainer.stop_training()
        self.assertTrue(not self.trainer.is_running)

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

    def test_modes_gpu(self):
        self._test_modes(True)