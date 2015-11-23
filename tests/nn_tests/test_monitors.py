import tempfile
import time
import unittest

import numpy as np
import numpy.testing as nptest
from chainer.functions import mean_squared_error
from chainer.link import Chain
from chainer.links import Linear
from chainer.optimizer import GradientClipping
from chainer.optimizers import SGD

from nt.nn import VariableInspector, SnapshotMonitor, LoggerMonitor, Trainer, \
    DataProvider
from nt.nn.data_fetchers import ArrayDataFetcher

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

    def forward_train(self, **kwargs):
        return self.forward(**kwargs)

    def forward_cv(self, **kwargs):
        return self.forward(**kwargs)


class LoggerMonitorTest(unittest.TestCase):
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
        self.trainer = Trainer(self.nn,
                               forward_fcn_tr=self.nn.forward_train,
                               forward_fcn_cv=self.nn.forward_cv,
                               data_provider_tr=self.tr_provider,
                               data_provider_cv=self.cv_provider,
                               optimizer=SGD(),
                               description='unittest',
                               data_dir=tempfile.mkdtemp(),
                               epochs=100,
                               use_gpu=False,
                               optimizer_hooks=hooks)
        g = self.trainer.get_computational_graph()
        self.monitor = LoggerMonitor(VariableInspector(-3, g), 'Input', True)
        self.trainer.add_tr_monitor(self.monitor)

    def test_logging(self):
        self.trainer.start_training()
        time.sleep(2)
        self.trainer.stop_training()
        self.assertGreater(len(self.monitor.logging_lists[0]), 0)


class SnapshotMonitorTest(unittest.TestCase):
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
        self.trainer = Trainer(self.nn,
                               forward_fcn_tr=self.nn.forward_train,
                               forward_fcn_cv=self.nn.forward_cv,
                               data_provider_tr=self.tr_provider,
                               data_provider_cv=self.cv_provider,
                               optimizer=SGD(),
                               description='unittest',
                               data_dir=tempfile.mkdtemp(),
                               epochs=100,
                               use_gpu=False,
                               optimizer_hooks=hooks)
        g = self.trainer.get_computational_graph()
        self.monitor = SnapshotMonitor(VariableInspector(-3, g), 'Input', True)
        self.trainer.add_tr_monitor(self.monitor)

    def test_logging(self):
        self.trainer.start_training()
        time.sleep(2)
        # Monitor is for training so we skip CV for this test
        while self.trainer.training_status.current_mode == 'Cross-validation':
            pass
        self.trainer.stop_training()
        nptest.assert_array_equal(
            self.monitor.logging_lists[0], self.trainer.current_batch)
