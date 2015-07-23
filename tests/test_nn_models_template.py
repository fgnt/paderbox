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


def forward_train(self, net):
    h1 = chainer.functions.relu(net.layers.l1(net.inputs.x))
    y = chainer.functions.relu(net.layers.l2(h1))
    loss = chainer.functions.mean_squared_error(net.inputs.target, y)
    net.outputs.y, net.outputs.h1 = y, h1
    return loss

def forward_decode(self, net):
    h1 = chainer.functions.relu(net.layers.l1(net.inputs.x))
    y = chainer.functions.relu(net.layers.l2(h1))
    net.outputs.h1 = h1
    return y

to_gpu = chainer.cuda.to_gpu
to_cpu = chainer.cuda.to_cpu

class NeuralNetworkTest(unittest.TestCase):

    def setUp(self):
        outputs = tuple('y h1'.split())
        inputs = tuple('x target'.split())
        self.nn = NeuralNetwork(inputs=inputs, outputs=outputs)
        self.nn.layers.l1 = chainer.functions.Linear(10, 10)
        self.nn.layers.l2 = chainer.functions.Linear(10, 5)

    def test_setup(self):
        self.assertTrue(self.nn.inputs.x is None)
        self.assertTrue(self.nn.inputs.target is None)
        self.assertTrue(self.nn.outputs.y is None)
        self.assertTrue(self.nn.outputs.h1 is None)

    def test_to_gpu(self):
        self.nn.to_gpu()
        self.assertTrue(self.nn.inputs.x is None)
        self.assertTrue(self.nn.inputs.target is None)
        self.assertTrue(self.nn.outputs.y is None)
        self.assertTrue(self.nn.outputs.h1 is None)
        data = numpy.random.uniform(0, 1, (2, 2))
        self.nn.inputs.x = data
        self.assertTrue(isinstance(self.nn.inputs.__dict__['x'],
                                   chainer.Variable))
        self.assertTrue(isinstance(self.nn.inputs.__dict__['x'].data,
                                   chainer.cuda.GPUArray))
        self.assertTrue(isinstance(self.nn.inputs.x, numpy.ndarray))
        numpy.testing.assert_array_equal(data, self.nn.inputs.x)

    def _test_raise_on_unknown_get(self):
        def get_a():
            return self.nn.inputs.a
        self.assertRaises(KeyError, get_a)

        def get_b():
            return self.nn.inputs.b
        self.assertRaises(KeyError, get_b)

    def test_raise_on_unknown_get_cpu(self):
        self._test_raise_on_unknown_get()

    def test_raise_on_unknown_get_gpu(self):
        self.nn.to_gpu()
        self._test_raise_on_unknown_get()

    # def test_print_variable(self):
    #     with patch('sys.stdout', new=StringIO()) as fakeOut:
    #         self.nn.print_config()
    #         output = fakeOut.getvalue().strip()
    #         self.assertEqual(output, 'p1.....val_test\np2.....val2')

    # def test_fw_train_cpu(self):
    #     data = nt.utils.Container()
    #     data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
    #     data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
    #     loss = self.nn.forward_train(data)
    #     self.assertTrue(isinstance(loss, chainer.Variable))
    #     self.assertTrue(isinstance(self.nn.data.y, chainer.Variable))
    #     self.assertTrue(isinstance(self.nn.data.h1, chainer.Variable))
    #
    # def test_save_load_cpu_cpu(self):
    #     data = nt.utils.Container()
    #     data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
    #     data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
    #     loss = self.nn.forward_train(data).data
    #     y = self.nn.data.y.data
    #     h1 = self.nn.data.h1.data
    #     self.nn.save('test_save_file')
    #     self.nn = DummyNetwork().load('test_save_file')
    #     self.assertEqual(self.nn.config['p1'], 'val_test')
    #     numpy.testing.assert_equal(self.nn.forward_train(data).data, loss)
    #     numpy.testing.assert_equal(self.nn.data.y.data, y)
    #     numpy.testing.assert_equal(self.nn.data.h1.data, h1)
    #     os.remove('test_save_file')

    # def test_to_gpu(self):
    #     data = nt.utils.Container()
    #     data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
    #     data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
    #     data.x = chainer.cuda.to_gpu(data.x)
    #     data.target = chainer.cuda.to_gpu(data.target)
    #     self.nn.to_gpu()
    #     self.assertTrue(self.nn.is_on_gpu)
    #     self.assertTrue(isinstance(self.nn.model.l1.W, chainer.cuda.GPUArray))
    #     loss = self.nn.forward_train(data)
    #     self.assertTrue(isinstance(loss.data, chainer.cuda.GPUArray))

    # def test_save_gpu(self):
    #     data = nt.utils.Container()
    #     data.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
    #     data.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
    #     data.x = chainer.cuda.to_gpu(data.x)
    #     data.target = chainer.cuda.to_gpu(data.target)
    #     self.nn.to_gpu()
    #     loss = self.nn.forward_train(data).data
    #     y, h1 = self.nn.data.y.data, self.nn.data.h1.data
    #     self.nn.save('test_save_file')
    #     self.nn = DummyNetwork().load('test_save_file')
    #     self.assertEqual(self.nn.config['p1'], 'val_test')
    #     data.x = to_cpu(data.x)
    #     data.target = to_cpu(data.target)
    #     numpy.testing.assert_almost_equal(self.nn.forward_train(data).data,
    #                                       to_cpu(loss), decimal=4)
    #     numpy.testing.assert_almost_equal(self.nn.data.y.data, to_cpu(y),
    #                                       decimal=4)
    #     numpy.testing.assert_almost_equal(self.nn.data.h1.data, to_cpu(h1),
    #                                       decimal=4)
    #     os.remove('test_save_file')