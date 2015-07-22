from nt.nn.models.template import NeuralNetwork
import unittest
import chainer
from io import StringIO
from unittest.mock import patch
import nt.utils
import numpy
import numpy.testing
import os
import chainer.cuda

if chainer.cuda.available:
    chainer.cuda.init()

class DummyNetwork(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self)

    def setup(self, p1='val1', p2='val2'):
        self.config['p1'] = p1
        self.config['p2'] = p2
        self.model = chainer.FunctionSet(
            l1=chainer.functions.Linear(10, 10),
            l2=chainer.functions.Linear(10, 5)
        )
        return self

    def forward_train(self, data, cv=False):
        x = chainer.Variable(data.x, volatile=cv)
        target = chainer.Variable(data.target, volatile=cv)
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

to_gpu = chainer.cuda.to_gpu
to_cpu = chainer.cuda.to_cpu

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

    def test_fw_train_cpu(self):
        data = nt.utils.Container()
        data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        loss = self.nn.forward_train(data)
        self.assertTrue(isinstance(loss, chainer.Variable))
        self.assertTrue(isinstance(self.nn.data.y, chainer.Variable))
        self.assertTrue(isinstance(self.nn.data.h1, chainer.Variable))

    def test_save_load_cpu_cpu(self):
        data = nt.utils.Container()
        data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        loss = self.nn.forward_train(data).data
        y = self.nn.data.y.data
        h1 = self.nn.data.h1.data
        self.nn.save('test_save_file')
        self.nn = DummyNetwork().load('test_save_file')
        self.assertEqual(self.nn.config['p1'], 'val_test')
        numpy.testing.assert_equal(self.nn.forward_train(data).data, loss)
        numpy.testing.assert_equal(self.nn.data.y.data, y)
        numpy.testing.assert_equal(self.nn.data.h1.data, h1)
        os.remove('test_save_file')

    def test_to_gpu(self):
        data = nt.utils.Container()
        data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        data.x = chainer.cuda.to_gpu(data.x)
        data.target = chainer.cuda.to_gpu(data.target)
        self.nn.to_gpu()
        self.assertTrue(self.nn.is_on_gpu)
        self.assertTrue(isinstance(self.nn.model.l1.W, chainer.cuda.GPUArray))
        loss = self.nn.forward_train(data)
        self.assertTrue(isinstance(loss.data, chainer.cuda.GPUArray))

    def test_save_gpu(self):
        data = nt.utils.Container()
        data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        data.x = chainer.cuda.to_gpu(data.x)
        data.target = chainer.cuda.to_gpu(data.target)
        self.nn.to_gpu()
        loss = self.nn.forward_train(data).data
        y, h1 = self.nn.data.y.data, self.nn.data.h1.data
        self.nn.save('test_save_file')
        self.nn = DummyNetwork().load('test_save_file')
        self.assertEqual(self.nn.config['p1'], 'val_test')
        data.x = to_cpu(data.x)
        data.target = to_cpu(data.target)
        numpy.testing.assert_almost_equal(self.nn.forward_train(data).data,
                                          to_cpu(loss), decimal=4)
        numpy.testing.assert_almost_equal(self.nn.data.y.data, to_cpu(y),
                                          decimal=4)
        numpy.testing.assert_almost_equal(self.nn.data.h1.data, to_cpu(h1),
                                          decimal=4)
        os.remove('test_save_file')