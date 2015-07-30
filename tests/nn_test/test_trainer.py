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
                               grad_clip=5)

    def _forward(self):
        h0 = F.sigmoid(self.nn.layers.l1(self.nn.inputs.i))
        y = self.nn.layers.l2(h0)
        self.nn.outputs.l = F.mean_squared_error(y, self.nn.inputs.t)
        self.nn.outputs.h0 = h0
        self.nn.outputs.y = y

    def test_init(self):
        self.assertTrue(os.path.exists(self.trainer.data_dir))

    def test_run_stop(self):
        self.trainer.start_training()
        self.assertTrue(self.trainer.training_thread.is_alive())
        time.sleep(2)
        self.assertTrue(self.trainer.training_thread.is_alive())
        self.trainer.stop_training()
        self.assertTrue(not self.trainer.training_thread.is_alive())

    def test_request(self):
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
