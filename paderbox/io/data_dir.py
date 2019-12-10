"""

>>> from pathlib import Path
>>> p = Path('/') / 'net' # /storage/python_unittest_data
>>> p
PosixPath('/net')
>>> p = p / 'storage'
>>> p
PosixPath('/net/storage')
>>> str(p)
'/net/storage'

"""

import os
from pathlib import Path
import urllib.request as url


def _get_path(environment_name, default):
    return Path(os.getenv(environment_name, default)).expanduser()


database_jsons = _get_path(
    'NT_DATABASE_JSONS_DIR',
    '/net/vol/jenkins/jsons'
)
db_dir = _get_path(
    'NT_DB_DIR',
    '/net/db'
)
fast_db_dir = _get_path(
    'NT_FAST_DB_DIR',
    '/net/fastdb'
)
ssd_db_dir = _get_path(
    'NT_SSD_DB_DIR',
    '/net/ssddb'
)
# testing = get_test_data(
# )
kaldi_root = _get_path(
    'KALDI_ROOT',
    '/path/to/proper/kaldi_root/check/in/getting/started/how/to/set/it'
)
matlab_toolbox = _get_path(
    'MATLAB_TOOLBOX_DIR',
    '/net/ssd/software/matlab_toolbox'
)
matlab_r2015a = _get_path(
    'MATLAB_R2015a',
    '/net/ssd/software/MATLAB/R2015a'
)
matlab_license = _get_path(
    'MATLAB_LICENSE',
    '/opt/MATLAB/R2016b_studis/licenses/network.lic'
)

ami = _get_path(
    'NT_AMI_DIR',
    db_dir / 'ami'
)
audioset = _get_path(
    'NT_AUDIOSET_DIR',
    ssd_db_dir / 'AudioSet'
)
dcase_2016_task_2 = _get_path(
    'NT_DCASE_2016_TASK_2_DIR',
    db_dir / 'DCASE2016'
)
dcase_2017_task_3 = _get_path(
    'NT_DCASE_2017_TASK_3_DIR',
    db_dir / 'DCASE2017' / 'Task3'
)
dcase_2017_task_4 = _get_path(
    'NT_DCASE_2017_TASK_4_DIR',
    ssd_db_dir / 'DCASE2017' / 'Task4'
)
dcase_2018_task_5 = _get_path(
    'NT_DCASE_2018_TASK_5_DIR',
    fast_db_dir / 'DCASE2018' / 'Task5'
)
dcase_2019_task_2 = _get_path(
    'NT_DCASE_2019_TASK_2_DIR',
    ssd_db_dir / 'DCASE2019' / 'Task2'
)
sins = _get_path(
    'NT_SINS_DIR',
    ssd_db_dir / 'SINS'
)
timit = _get_path(
    'NT_TIMIT_DIR',
    db_dir / 'timit'
)
tidigits = _get_path(
    'NT_TIDIGITS_DIR',
    db_dir / 'tidigits'
)
chime_3 = _get_path(
    'NT_CHIME_3_DIR',
    fast_db_dir / 'chime3'
)
chime_4 = _get_path(
    'NT_CHIME_4_DIR',
    fast_db_dir / 'chime4'
)
chime_5 = _get_path(
    'NT_CHIME_5_DIR',
    fast_db_dir / 'chime5'
)
merl_mixtures = _get_path(
    'NT_MERL_MIXTURES_DIR',
    fast_db_dir / 'merl_speaker_mixtures'
)
wham = _get_path(
    'NT_WHAM_DIR',
    db_dir / 'wham'
)
german_speechdata = _get_path(
    'NT_GERMAN_SPEECHDATA_DIR',
    '/net/storage/jheymann/speech_db/german-speechdata-package-v2/'
)
noisex92 = _get_path(
    'NT_NoiseX_92_DIR',
    db_dir / 'NoiseX_92'
)
reverb = _get_path(
    'NT_REVERB_DIR',
    fast_db_dir / 'reverb'
)
wsj = _get_path(
    'NT_WSJ_DIR',
    fast_db_dir / 'wsj'
)
wsj_voicehome = _get_path(
    'NT_WSJ_VOICEHOME_DIR',
    db_dir / 'voicehome_v2'
)
wsj_corrected_paths = _get_path(
    'NT_WSJ_DIR',
    db_dir / 'wsj_corrected_paths'
)
wsj_8k = _get_path(
    'NT_WSJ_8K_DIR',
    fast_db_dir / 'wsj_8k'
)
wsjcam0 = _get_path(
    'NT_WSJCAM0_DIR',
    db_dir / 'wsjcam0'
)
wsj_bss = _get_path(
    'NT_WSJ_BSS_DIR',
    fast_db_dir / 'wsj_bss'
)

# This language model was used in our CHIME 4 submission.
language_model = _get_path(
    'LANGUAGE_MODEL',
    '/net/vol/ldrude/projects/2016/2016-05-10_lm'
)

wsj_mc = _get_path(
    'NT_WSJ_MC_DIR',
    db_dir / 'wsj_mc_complete'
)
librispeech = _get_path(
    'NT_LIBRISPEECH_DIR',
    db_dir / 'LibriSpeech'
)
ham_radio_librispeech = _get_path(
    'NT_HAM_RADIO_LIBRISPEECH_DIR',
    db_dir / '/net/db/ham'
)
mird = _get_path(
    'NT_MIRD_DIR',
    db_dir / 'Multichannel_Impulse_Response_Database'
)

dipco = _get_path(
    'NT_DIPCO_DIR',
    Path('/net/vol/boeddeker/share/DiPCo/DiPCo')
    # db_dir / 'DiPCo'
)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
