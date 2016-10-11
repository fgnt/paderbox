"""
This module simply wraps all numpy.testing functions and provides additional
assertions for our cases.
"""
from nt.testing.module_asserts import *
from numpy.testing import *
from chainer.testing.condition import retry
import nt.testing.attr
