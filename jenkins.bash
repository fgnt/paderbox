#!/usr/bin/env bash

# include common stuff (installation of toolbox, paths, traps, nice level...)
source "`dirname "$0"`/jenkins_common.bash"

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation
# pytest ignores all evaluation tests, generate_data_file.py and  import_test.py due to errors

# nosetests -a '!matlab' --with-xunit --with-coverage --cover-package=paderbox -v -w "${TOOLBOX}/tests" # --processes=-1

pytest --junitxml="test_results.xml" --cov=paderbox --doctest-modules   \
       --doctest-continue-on-failure --cov-report term -v "${TOOLBOX}/tests/" \
       -k "not matlab"

# Use as many processes as you have cores: --processes=-1

# Export coverage
python -m coverage xml --include="${TOOLBOX}/paderbox*"

# Pylint tests
pylint --rcfile="${TOOLBOX}/pylint.cfg" -f parseable paderbox > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes paderbox
