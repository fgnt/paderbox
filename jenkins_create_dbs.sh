#!/usr/bin/env bash

renice -n 20 $$

# set a prefix for each cmd
green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

# Force Exit 0
# trap 'exit 0' EXIT SIGINT SIGTERM EXIT

# Set Paths
CUDA_PATH=/usr/local/cuda
LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH}
PATH=$CUDA_PATH/bin:$PATH
export PATH
export LD_LIBRARY_PATH
source activate py35

# Use a pseudo virtualenv, http://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
mkdir -p venv
export PYTHONUSERBASE=$(readlink -m venv)


# Refresh toolbox
pip install  --quiet --user -e .

BUILD_DIR=build/database_jsons
DST_DIR=/net/storage/database_jsons

mkdir -p $BUILD_DIR
cd $BUILD_DIR

INFO=info.txt

echo "" > $INFO
# echo "" > /net/storage/database_jsons/BUILDING_NOW

# Create info
echo "Create on `date`" >> $INFO
echo "Git revision `git rev-parse HEAD`" >> $INFO

# Create and copy databases

python -m nt.database.tidigits.JSON_conv_tidigits

python -m nt.database.timit.create_json

python -m nt.database.wsj.create_json

python -m nt.database.merl_mixtures.merl_speaker_mixtures --sample_rate wav8k --min_max min --json_path 'merl_speaker_mixtures_min_8k.json'
python -m nt.database.merl_mixtures.merl_speaker_mixtures --sample_rate wav8k --min_max max --json_path 'merl_speaker_mixtures_max_8k.json'
python -m nt.database.merl_mixtures.merl_speaker_mixtures --sample_rate wav16k --min_max min --json_path 'merl_speaker_mixtures_min_16k.json'
python -m nt.database.merl_mixtures.merl_speaker_mixtures --sample_rate wav16k --min_max max --json_path 'merl_speaker_mixtures_max_16k.json'

python -m nt.database.reverb.gen_config
python -m nt.database.reverb.process_db

python -m nt.database.german_speechdata_package_v2.database_german_speechdata_package_v2

python -m nt.database.noisex_92.database_NoiseX_92

python -m nt.database.chime.create_background_json

python -m nt.database.chime.gen_config
python -m nt.database.chime.process_db --json

# Uninstall packages
pip uninstall --quiet --yes nt

# rm /net/storage/database_jsons/BUILDING_NOW

# exit # uncomment for testing

# -n, --dry-run    perform a trial run with no changes made
rsync --itemize-changes --archive --backup --human-readable --verbose \
--partial --progress --stats  *.json ${DST_DIR} \
--backup-dir=${DST_DIR}/backup/$(date +%Y_%m_%d_%T) \
--checksum

mv $INFO ${DST_DIR}