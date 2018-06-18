from pathlib import Path
from shutil import copytree, copyfile
from nt.utils.process_caller import run_process
from nt.database.chime5 import kaldi_to_nt_example_id
import os
import stat
import sacred
import gzip
from nt.database import JsonDatabase, keys
from chime5.scripts.kaldi import check_dest_dir, ORG_DIR
mapping_json = '/net/vol/jensheit/test/mapping.json'
db = JsonDatabase(mapping_json)
ex = sacred.Experiment('Chime5 train HMM GMM on array')

class MapFunKaldiInear2Array:
    def __init__(self, num_arrays=6, num_channels=4,
                 org_dir=ORG_DIR / 'data' / 'train_uall'):
        self.num_arrays = num_arrays
        self.num_channels = num_channels
        print(org_dir)
        text = (org_dir / 'text').open('r').readlines()
        self.ids = [line.split(' ')[0] for line in text]
        self.iterator = db.get_iterator_by_names('train')

    def __call__(self, example_id):
        nt_id = kaldi_to_nt_example_id(example_id)
        id_dict = self.iterator[nt_id]
        array_ids = (id_dict[keys.OBSERVATION][f'U0{array}'][channel]
                     for channel in range(self.num_channels)
                     for array in range(1, self.num_arrays + 1)
                     if not array == 3)
        return array_ids

    # def check_ids(self, array_ids):
    #     for _id in array_ids:
    #         if not _id in self.ids:
    #             if not 'U03' in _id:
    #                 print(f'{_id} was not found in default kaldi ids')
    #             continue
    #         yield _id

def multiply_alignments(org_ali_dir, out_ali_dir,
                        id_map_fun: MapFunKaldiInear2Array()):
    assert not out_ali_dir.exists() or len(list(org_ali_dir.glob('*.gz'))) == 0,\
        f'The destination dir {out_ali_dir} already exists, please choose a' \
        f' different dir'

    copytree(str(org_ali_dir), str(out_ali_dir), symlinks=True)
    for files in org_ali_dir.glob('*.gz'):
        id_list = list()
        ali_decompressed = gzip.decompress(files.open('rb').read())
        for ex in ali_decompressed.decode().split('\n'):
            if ex == '':
                break
            utt, utt_ali = ex.split(' ', maxsplit=1)
            id_list += [f'{new_utt_id} {utt_ali}'
                        for new_utt_id in id_map_fun(utt)]
            (out_ali_dir / files.name).write_bytes(gzip.compress('\n'.join(id_list).encode(), compresslevel=1))

def get_files(base_dir, train_set, dev_set, org_dir=ORG_DIR):
    if not (base_dir / 'data' / train_set).exists():
        copytree(str(org_dir / 'data' / train_set),
                 str(base_dir / 'data' / train_set))
    if not (base_dir / 'data' / dev_set).exists():
        copytree(str(org_dir / 'data' / dev_set),
                 str(base_dir / 'data' / dev_set))
    if not (base_dir / 'get_array_model.bash').exists():
        file = str(base_dir / 'get_model.bash')
        copyfile(str(org_dir / 'get_array_model.bash'), file)
        st = os.stat(file)
        os.chmod(file, st.st_mode | stat.S_IEXEC)


def calculate_alignments(base_dir, hmm_dir, train_set,
                         org_dir, ali_dir):
    if not (base_dir / 'data' / train_set).exists():
        copytree(str(org_dir / 'data' / train_set),
                 str(base_dir / 'data' / train_set))
    if not (base_dir / 'data' / train_set / 'feats.scp').exists():
        run_process([
            f'{base_dir}/steps/make_mfcc.sh', '--nj', '20', '--cmd', 'run.pl',
            f'data/{train_set}', f'exp/make_mfcc/{train_set}', 'mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )
    if not (base_dir / 'data' / train_set / 'cmvn.scp').exists():
        run_process([
            f'{base_dir}/steps/compute_cmvn_stats.sh',
            f'data/{train_set}', f'exp/make_mfcc/{train_set}', 'mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )
    run_process([
        f'{base_dir}/utils/fix_data_dir.sh', f'data/{train_set}'],
        cwd=str(base_dir), stdout=None, stderr=None
    )

    run_process([
        f'{base_dir}/steps/align_si.sh', '--nj', '16', '--cmd', 'run.pl',
        f'data/{train_set}', 'data/lang',
        f'exp/{hmm_dir}', str(ali_dir)],
        cwd=str(base_dir), stdout=None, stderr=None
    )


def train_hmm_array(base_dir: Path, train_set='train_uall',
                    dev_set='dev_beamformit_ref',  org_hmm='tri3_worn',
                    org_train_set='train_worn', org_dir=ORG_DIR,
                    num_arrays=6, num_channels=4):
    get_files(base_dir, train_set, dev_set, org_dir)
    check_dest_dir(base_dir)
    out_ali_dir = base_dir / 'exp' / f'{org_hmm}_{train_set}_ali'
    if not out_ali_dir.exists() or len(list(out_ali_dir.glob('*.gz'))) == 0:
        org_ali_dir = base_dir / 'exp' / f'{org_hmm}_{org_train_set}_ali'
        print(f'{out_ali_dir} not found, creating it from {org_ali_dir}')
        if not org_ali_dir.exists():
            print(f'{org_ali_dir} not found, calculating alignments')
            calculate_alignments(base_dir, org_hmm, org_train_set,
                                 org_dir, org_ali_dir)
        assert len(list(org_ali_dir.glob('*.gz'))) != 0, f'No alignments found' \
                                                         f'in org_dir {org_ali_dir}'
        map_fun = MapFunKaldiInear2Array(num_arrays, num_channels,
                                         org_dir / 'data' / train_set)
        multiply_alignments(org_ali_dir, out_ali_dir, map_fun)
    run_process([
        f'{base_dir}/get_array_model.bash',
        '--train_set', f'{train_set}',
        '--dev_sets', f'{dev_set}',
        '--org_hmm', f'{org_hmm}'],
        cwd=str(base_dir),
        stdout=None, stderr=None
    )


@ex.config
def default():
    train_set = 'train_uall'
    dev_set = 'dev_beamformit_ref'
    org_hmm = 'tri3_worn'
    org_train_set = 'train_worn'
    org_dir = ORG_DIR
    num_arrays = 6
    num_channels = 4


@ex.automain
def run(_config):
    assert len(ex.current_run.observers) > 0, (
        'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
    )
    base_dir = ex.current_run.observers[0].basedir

    train_hmm_array(Path(base_dir), train_set=_config['train_set'],
                    dev_set=_config['dev_set'], org_hmm=_config['org_hmm'],
                    org_train_set=_config['org_train_set'],
                    org_dir=_config['org_dir'],
                    num_arrays=_config['num_arrays'],
                    num_channels=_config['num_channels'])