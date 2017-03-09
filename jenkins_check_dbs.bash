#!/usr/bin/env bash

# include common stuff (installation of toolbox and chainer, paths, traps, nice level...)
source "`dirname "$0"`/jenkins_common.bash"

# execute test and create xml for jenkins
nosetests --with-xunit -s "${TOOLBOX}/check_json_wavs.py"

tear_down