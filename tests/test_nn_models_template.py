from nt.nn.models.template import NeuralNetwork
import unittest
import chainer
from io import StringIO
from unittest.mock import patch

class DummyNetwork(NeuralNetwork):
    def setup(self, p1='val1', p2='val2'):
        NeuralNetwork.__init__(self)
        self.config['p1'] = p1
        self.config['p2'] = p2
        self.model = chainer.FunctionSet(
            l1=chainer.functions.Linear(10, 10),
            l2=chainer.functions.Linear(10, 5)
        )
        return self

    def forward_train(self, data, cv=False):
        x = chainer.Variable(data.x)
        target = chainer.Variable(data.target)
        h1 = chainer.functions.relu(self.model.l1(x))
        y = chainer.functions.relu(self.model.l2(h1))
        self.loss = chainer.functions.mean_squared_error(target, y)
        self.data.y, self.data.h1 = y, h1
        return self.loss

    def forward_decode(self, data):
        x = chainer.Variable(data.x, False)
        h1 = chainer.functions.relu(self.model.l1(x))
        y = chainer.functions.relu(self.model.l2(x))
        self.data.h1 = h1
        return y

class NeuralNetworkTest(unittest.TestCase):

    def setUp(self):
        self.nn = DummyNetwork().setup(p1='val_test')

    def test_setup(self):
        self.assertEqual(self.nn.config['p1'], 'val_test')

    def test_print_variable(self):
        with patch('sys.stdout', new=StringIO()) as fakeOut:
            self.nn.print_config()
            output = fakeOut.getvalue().strip()
            self.assertEqual(output, 'p1.....val_test\np2.....val2')
