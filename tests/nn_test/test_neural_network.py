from nt.nn.models.template import *
import unittest
import chainer
from io import StringIO
from unittest.mock import patch
import nt.utils
import numpy
import numpy.testing
import os
import chainer.cuda
import pickle

# if chainer.cuda.available:
#     chainer.cuda.init()


def forward_train(nn):
    h1 = chainer.functions.relu(nn.layers.l1(nn.inputs.x))
    y = chainer.functions.relu(nn.layers.l2(h1))
    loss = chainer.functions.mean_squared_error(nn.inputs.target, y)
    nn.outputs.y, nn.outputs.h1 = y, h1
    return loss

def forward_decode(nn):
    h1 = chainer.functions.relu(nn.layers.l1(nn.inputs.x))
    y = chainer.functions.relu(nn.layers.l2(h1))
    nn.outputs.h1 = h1
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
        self.default = numpy.zeros((1, 1))

    def test_setup(self):
        self.assertTrue(isinstance(self.nn.inputs.x, chainer.Variable))
        numpy.testing.assert_equal(self.nn.inputs.x.data, self.default)
        self.assertTrue(isinstance(self.nn.inputs.target, chainer.Variable))
        numpy.testing.assert_equal(self.nn.inputs.target.data, self.default)
        self.assertTrue(isinstance(self.nn.outputs.y.num, numpy.ndarray))
        numpy.testing.assert_equal(self.nn.outputs.y.num, self.default)
        self.assertTrue(isinstance(self.nn.outputs.h1.num, numpy.ndarray))
        numpy.testing.assert_equal(self.nn.outputs.h1.num, self.default)

    def test_to_gpu(self):
        self.nn.to_gpu()
        self.assertTrue(isinstance(self.nn.inputs,
                                   GpuInputContainer))
        data = numpy.random.uniform(0, 1, (2, 2))
        self.nn.inputs.x = data
        self.assertTrue(isinstance(self.nn.inputs.x,
                                   chainer.Variable))
        self.assertTrue(isinstance(self.nn.inputs.x.data,
                                   chainer.cuda.GPUArray))

        self.assertTrue(isinstance(self.nn.outputs.y.num, numpy.ndarray))
        numpy.testing.assert_array_equal(self.default, self.nn.outputs.y.num)

    def _test_raise_on_unknown_get(self):
        def get_a():
            return self.nn.inputs.a
        self.assertRaises(AttributeError, get_a)

        def get_b():
            return self.nn.outputs.b
        self.assertRaises(AttributeError, get_b)

    def test_raise_on_unknown_get_cpu(self):
        self._test_raise_on_unknown_get()

    def test_raise_on_unknown_get_gpu(self):
        self.nn.to_gpu()
        self._test_raise_on_unknown_get()

    def test_fw_train_cpu(self):
        self.nn.inputs.x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        self.nn.inputs.target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        loss = forward_train(self.nn)
        self.assertTrue(isinstance(loss, chainer.Variable))
        self.assertTrue(isinstance(self.nn.outputs.y, chainer.Variable))
        self.assertTrue(isinstance(self.nn.outputs.h1, chainer.Variable))

    def _test_save_load(self, first_gpu=False, second_gpu=False):
        x = numpy.random.uniform(-1, 1, (20, 10)).astype(numpy.float32)
        target = numpy.random.uniform(-1, 1, (20, 5)).astype(numpy.float32)
        if first_gpu:
            self.nn.to_gpu()
        self.nn.set_inputs(x=x, target=target)
        loss = forward_train(self.nn).num
        y = self.nn.outputs.y.num
        h1 = self.nn.outputs.h1.num
        self.nn.save('test_save_file')
        self.nn = NeuralNetwork().load('test_save_file')
        if second_gpu:
            self.nn.to_gpu()
        self.nn.set_inputs(x=x, target=target)
        numpy.testing.assert_almost_equal(forward_train(self.nn).num, loss,
                                          decimal=3)
        numpy.testing.assert_almost_equal(self.nn.outputs.y.num, y, decimal=3)
        numpy.testing.assert_almost_equal(self.nn.outputs.h1.num, h1, decimal=3)
        os.remove('test_save_file')

    def test_save_load_cpu_cpu(self):
        self._test_save_load(False, False)

    def test_save_load_gpu_cpu(self):
        self._test_save_load(True, False)

    def test_save_load_cpu_gpu(self):
        self._test_save_load(False, True)

    def test_save_load_gpu_gpu(self):
        self._test_save_load(True, True)

    def test_to_gpu_error(self):
        self.nn.to_gpu()
