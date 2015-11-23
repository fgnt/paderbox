from nt.nn.monitoring import VariableInspector,\
    VariableGradientInspector,\
    BatchInspector
import unittest
import numpy as np
import numpy.testing as nptest
from chainer.link import Chain
from chainer.functions import mean_squared_error
from chainer.links import Linear
from chainer import Variable
from chainer import computational_graph
from nt.nn import DataProvider
from nt.nn.data_fetchers import ArrayDataFetcher

B = 10
A = 5

class DummyNetwork(Chain):

    def __init__(self):
        super(DummyNetwork, self).__init__(
            l1 = Linear(5, 3),
            l2 = Linear(3, 5)
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


class TestVariableInspector(unittest.TestCase):

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
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        loss = self.nn.forward_train(**batch_input)
        cg = computational_graph.build_computational_graph([loss])
        self.inspector = VariableInspector(-3, cg)

    def test_get_data(self):
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        loss = self.nn.forward_train(**batch_input)
        cg = computational_graph.build_computational_graph([loss])
        data = self.inspector.get_data(cg, batch)
        nptest.assert_array_equal(batch['i'], data)


class TestVariableGradientInspector(unittest.TestCase):

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
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        loss = self.nn.forward_train(**batch_input)
        cg = computational_graph.build_computational_graph([loss])
        self.inspector = VariableGradientInspector(-3, cg)

    def test_get_data(self):
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        loss = self.nn.forward_train(**batch_input)
        loss.backward()
        cg = computational_graph.build_computational_graph([loss])
        data = self.inspector.get_data(cg, batch)
        nptest.assert_array_equal(batch['i'].shape, data.shape)


class TestBatchInspector(unittest.TestCase):

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
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        self.inspector = BatchInspector('i', None, batch_input)

    def test_get_data(self):
        batch = self.tr_provider.test_run()
        batch_input = {k: Variable(v, name=k) for k,v in batch.items()}
        loss = self.nn.forward_train(**batch_input)
        cg = computational_graph.build_computational_graph([loss])
        data = self.inspector.get_data(cg, batch_input)
        nptest.assert_array_equal(batch['i'], data)
