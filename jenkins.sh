#!/usr/bin/env bash

# Set Paths
CUDA_PATH=/usr/local/cuda-6.5
LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
PATH=/net/ssd/software/anaconda/envs/py3k_jenkins/bin:$CUDA_PATH/bin:$PATH
export PATH
export LD_LIBRARY_PATH

# Refresh toolbox
/usr/bin/yes | pip uninstall nt || true
/usr/bin/yes | pip install --user . || true

# Update chainer
/usr/bin/yes | pip install --user --upgrade ./chainer/ || true

# Unittets
nosetests --with-xunit --all-modules --with-coverage --cover-package=nt || true

# Export coverage
python -m coverage xml --include=nt* || true

# Pylint tests
/net/ssd/software/anaconda/envs/py3k_jenkins/bin/pylint --rcfile=pilint.cfg -f parseable nt || true
make --directory=doc html || true

# Store pip packages
pip freeze > pip.txt || true

# Uninstall packages
/usr/bin/yes | pip uninstall chainer || true
/usr/bin/yes | pip uninstall nt || true