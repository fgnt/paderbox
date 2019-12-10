
import pytest

"""
    see https://docs.pytest.org/en/latest/example/markers.html#mark-examples

    @paderbox.testing.attr.matlab
    def test_matlab():
        pass

    @paderbox.testing.attr.matlab
    def TestMatlab:
        pass

    pytest -v -m "not matlab"  # disable test_matlab and TestMatlab

"""

matlab = pytest.mark.matlab
