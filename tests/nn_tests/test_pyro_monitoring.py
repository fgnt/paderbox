import unittest

import numpy as np
from numpy.testing import assert_raises_regex
from nt import testing as tc
from chainer.variable import Variable
from nt.nn.pyro_monitoring import PyroMonitorServer
from nt.utils import AttrDict

# This is an extension to the ipynb notebook test

class LoggerMonitorTest(unittest.TestCase):

    def setUp(self):
        self.pyro_mon = PyroMonitorServer()

        self.tr_log_data_args = AttrDict(
            batch={'x': np.arange(4)},
            net_out={'y': np.arange(6)},
            computational_graph=AttrDict(nodes_dict={
                ('Var', 0): Variable(np.arange(5)),
            }),
            observation_indices=None,  # ToDo
            in_cv=False,
        )

        self.cv_log_data_args = AttrDict(
            batch={'x': np.arange(2)},
            net_out={'y': np.arange(3)},
            computational_graph=None,
            observation_indices=None,  # ToDo
            in_cv=True,
        )

    def test_log_data(self):
        self.pyro_mon.log_data(**self.tr_log_data_args)
        self.pyro_mon.log_data(**self.cv_log_data_args)

    def test_add_observer(self):
        self.pyro_mon.log_data(**self.tr_log_data_args)
        self.pyro_mon.log_data(**self.cv_log_data_args)

        with assert_raises_regex(KeyError, 'Wrong signal_identifier'):
            self.pyro_mon.add_observer('y', 'B', logging=True,
                                       allow_reset=False)

        self.pyro_mon.add_observer('y', 'O', logging=True, allow_reset=False)
        self.pyro_mon.add_observer('x', 'B', logging=True)
        self.pyro_mon.add_observer(('Var', 0), 'V', name='Var')

        with assert_raises_regex(
                ValueError, "origin_of_signal must be 'B', 'O', 'V' or 'F'"):
            self.pyro_mon.add_observer('y', 'X', logging=True,
                                       allow_reset=False)

        tc.assert_equal({'tr', 'cv'}, self.pyro_mon.data.keys())
        tc.assert_equal({'x', 'y', 'Var'}, self.pyro_mon.tr.keys())
        tc.assert_equal({'x', 'y'}, self.pyro_mon.cv.keys())

        # Without log

        tc.assert_equal(self.pyro_mon.tr, dict(
            y=[], x=[], Var=[],
        ))

        tc.assert_equal(self.pyro_mon.cv, dict(
            y=[], x=[],
        ))

        # One log

        self.pyro_mon.log_data(**self.tr_log_data_args)
        self.pyro_mon.log_data(**self.cv_log_data_args)

        tc.assert_equal(self.pyro_mon.tr, dict(
            y=[np.arange(6)], x=[np.arange(4)], Var=np.arange(5),
        ))
        tc.assert_equal(self.pyro_mon.cv, dict(
            y=[np.arange(3)], x=[np.arange(2)],
        ))

        # Two logs

        self.pyro_mon.log_data(**self.tr_log_data_args)
        self.pyro_mon.log_data(**self.cv_log_data_args)

        tc.assert_equal(self.pyro_mon.data, dict(
            tr=dict(y=[np.arange(6), np.arange(6)],
                    x=[np.arange(4), np.arange(4)],
                    Var=np.arange(5),),
            cv=dict(y=[np.arange(3), np.arange(3)],
                    x=[np.arange(2), np.arange(2)],),
        ))

        # Reset tr

        self.pyro_mon.next_epoch_tr()

        tc.assert_equal(self.pyro_mon.data, dict(
            tr=dict(y=[np.arange(6), np.arange(6)],
                    x=[],
                    Var=[], ),
            cv=dict(y=[np.arange(3), np.arange(3)],
                    x=[np.arange(2), np.arange(2)], ),
        ))

        # Reset cv

        self.pyro_mon.next_epoch_cv()

        tc.assert_equal(self.pyro_mon.data, dict(
            tr=dict(y=[np.arange(6), np.arange(6)],
                    x=[],
                    Var=[], ),
            cv=dict(y=[np.arange(3), np.arange(3)],
                    x=[], ),
        ))

        # hard reset

        self.pyro_mon.reset()

        tc.assert_equal(self.pyro_mon.data, dict(
            tr=dict(y=[],
                    x=[],
                    Var=[], ),
            cv=dict(y=[],
                    x=[], ),
        ))

        # drop one observer

        self.pyro_mon.drop_observer('x')

        tc.assert_equal({'y', 'Var'}, self.pyro_mon.tr.keys())
        tc.assert_equal({'y'}, self.pyro_mon.cv.keys())

        # drop all observer

        self.pyro_mon.drop_all_observer()

        assert 0 == len(self.pyro_mon.tr.keys())
        assert 0 == len(self.pyro_mon.cv.keys())
