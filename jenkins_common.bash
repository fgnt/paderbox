#!/usr/bin/env bash

renice -n 20 $$

# set a prefix for each cmd
green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

# Force Exit 0
trap 'exit 0' EXIT SIGINT SIGTERM

PYTHON_PATH=/net/software/python/2022_02/anaconda

source $PYTHON_PATH/bin/activate

# Use a pseudo virtualenv, http://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
mkdir -p venv
export PYTHONUSERBASE=$(readlink -m venv)

# paths
TOOLBOX="$(dirname $(readlink -f ${BASH_SOURCE[0]}))"

# Refresh files...
ls $PYTHON_PATH/lib/python3.6/lib-dynload/../../ > /dev/null

# Some debug information
env

# adds a KALDI_ROOT
source "${TOOLBOX}/bash/kaldi.bash"

# Refresh toolbox
pip uninstall --quiet --yes paderbox
pip show paderbox
pip install --quiet --user -e ${TOOLBOX}
pip show paderbox

