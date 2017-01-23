#!/usr/bin/env bash

# include common stuff (installation of toolbox and chainer, paths, traps, nice level...)
source "`dirname "$0"`/jenkins_common.bash"

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation

nosetests -a '!matlab' --with-xunit --with-coverage --cover-package=nt -v -w ${TOOLBOX} # --processes=-1
# Use as many processes as you have cores: --processes=-1

# Export coverage
python -m coverage xml --include=${TOOLBOX}/nt*

# Pylint tests
pylint --rcfile=${TOOLBOX}/pylint.cfg -f parseable nt > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

env

# Build documentation
make --directory=${TOOLBOX}/doc/source/auto_reference/ clean
make --directory=${TOOLBOX}/doc/source/auto_reference/
make --directory=${TOOLBOX}/doc clean
make --directory=${TOOLBOX}/doc html

# Store pip packages
pip freeze > pip.txt

# Uninstall packages
pip uninstall --quiet --yes chainer
pip uninstall --quiet --yes nt
