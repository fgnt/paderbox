
install:
	pip install --user -e .

cython:
	python3 setup.py build_ext --inplace

.PHONY: tests
tests:
	nosetests --with-xunit -v -w "tests"

clean:
	rm -f pyrirgen.cpp
	rm -r pyrirgen.html
	rm -r pyrirgen.cpython*
