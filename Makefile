

error:
	echo "Wrong arguments, see cat Makefile"

cython:
	python setup.py build_ext --inplace

install:
	pip install --user -e .
