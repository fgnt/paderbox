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

# Create info
echo "Create on `date`" >> $INFO
echo "Git revision `git rev-parse HEAD`" >> $INFO
echo "" >> $INFO
echo "" >> $INFO
echo "LOG:" >> $INFO

# Create and copy databases

echo 'Creating TIDIGITS json' || tee -a $INFO
python nt/database/tidigits/JSON_conv_tidigits.py || tee -a
cp tidigits.json /net/storage/database_jsons/tidigits.json
echo "" >> $INFO

echo 'Creating TIMIT json' || tee -a $INFO
python nt/database/timit/database_timit.py || tee -a
cp TIMIT.json /net/storage/database_jsons/timit.json
echo "" >> $INFO

echo 'Creating WSJ json' || tee -a $INFO
python nt/database/wsj/database_wsj.py || tee -a $INFO
cp wsj.json /net/storage/database_jsons/wsj.json
echo "" >> $INFO

echo 'Creating Reverb json' || tee -a $INFO
python nt/database/reverb/gen_config.py || tee -a $INFO
python nt/database/reverb/process_db.py || tee -a $INFO
cp reverb.json /net/storage/database_jsons/reverb.json
echo "" >> $INFO

echo 'Creating GERMAN json' || tee -a $INFO
python nt/database/german-speechdata-package-v2/database_german-speechdata-package-v2.py || tee -a $INFO
cp german-speechdata-package-v2.json /net/storage/database_jsons/german.json
echo "" >> $INFO


# Uninstall packages
/usr/bin/yes | pip uninstall nt || true
