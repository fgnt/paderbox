#!/usr/bin/env bash

# Set Paths
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
PATH=/net/ssd/software/anaconda/envs/py3k_jenkins/bin:$CUDA_PATH/bin:$PATH
export PATH
export LD_LIBRARY_PATH

# Refresh toolbox
/usr/bin/yes | pip install --user -e . || true

# Create and copy databases
python nt/database/timit/database_timit.py
cp nt/database/timit/timit.json /net/storage/database_jsons/timit.json

# Create info
echo "Create on `date`" >> /net/storage/database_jsons/info.txt
echo "Git revision `git rev-parse HEAD`" >> /net/storage/database_jsons/info.txt

# Uninstall packages
/usr/bin/yes | pip uninstall nt || true
