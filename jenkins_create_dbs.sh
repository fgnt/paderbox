#!/usr/bin/env bash

# Set Paths
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
PATH=/net/ssd/software/anaconda/envs/py3k_jenkins/bin:$CUDA_PATH/bin:$PATH
export PATH
export LD_LIBRARY_PATH

# Refresh toolbox
/usr/bin/yes | pip install --user -e . || true

INFO=/net/storage/database_jsons/info.txt

rm $INFO
echo "" > /net/storage/database_jsons/BUILDING_NOW

# Create info
echo "Create on `date`" >> $INFO
echo "Git revision `git rev-parse HEAD`" >> $INFO

# Create and copy databases

echo 'Creating TIDIGITS json'
python nt/database/tidigits/JSON_conv_tidigits.py
cp tidigits.json /net/storage/database_jsons/tidigits.json

echo 'Creating TIMIT json'
python nt/database/timit/database_timit.py
cp TIMIT.json /net/storage/database_jsons/timit.json

echo 'Creating WSJ json'
python nt/database/wsj/database_wsj.py
cp wsj.json /net/storage/database_jsons/wsj.json

echo 'Creating Reverb json'
cd nt/database/reverb
python gen_config.py
python process_db.py
cp reverb.json /net/storage/database_jsons/reverb.json
cd ../../..

echo 'Creating GERMAN json'
python nt/database/german_speechdata_package_v2/database_german_speechdata_package_v2.py
cp german_speechdata_package_v2.json /net/storage/database_jsons/german.json

# Uninstall packages
/usr/bin/yes | pip uninstall nt || true

rm /net/storage/database_jsons/BUILDING_NOW
