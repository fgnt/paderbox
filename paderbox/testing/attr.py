
from nose.plugins import attrib

"""
    see http://nose.readthedocs.io/en/latest/plugins/attrib.html

    @nt.testing.attr.matlab
    def test_matlab():
        pass

    @nt.testing.attr.matlab
    def TestMatlab:
        pass

    nosetests -a '!matlab'  # disable test_matlab and TestMatlab

"""

matlab = attrib.attr(matlab=True)
