#!/usr/bin/env bash

# include common stuff (installation of toolbox and chainer, paths, traps, nice level...)
source "`dirname "$0"`/jenkins_common.bash"

python check_json_wavs.py

# Uninstall packages
pip uninstall --quiet --yes nt
pip uninstall --quiet --yes chainer