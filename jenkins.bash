#!/usr/bin/env bash

# include common stuff (installation of toolbox, paths, traps, nice level...)
source "`dirname "$0"`/jenkins_common.bash"

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation

nosetests -a '!matlab' --with-xunit --with-coverage --cover-package=nt -v -w "${TOOLBOX}/tests" # --processes=-1
# Use as many processes as you have cores: --processes=-1

# Export coverage
python -m coverage xml --include="${TOOLBOX}/nt*"

# Pylint tests
pylint --rcfile="${TOOLBOX}/pylint.cfg" -f parseable nt > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

make --directory=./toolbox/doc/ clean
make --directory=./toolbox/doc/ html

pip freeze > pip.txt
pip uninstall --quiet --yes nt
