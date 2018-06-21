from pathlib import Path
import tqdm
from shutil import copytree, copyfile
from nt.utils.process_caller import run_process
import os
import stat
import sacred
import gzip
from nt.database import JsonDatabase, keys
from chime5.scripts.kaldi import check_dest_dir, ORG_DIR
from nt.database.chime5 import Chime5, adjust_start_end

mapping_json = '/net/vol/jensheit/test/mapping.json'
db = JsonDatabase(mapping_json)
iterator = db.get_iterator_by_names('train')

chime5_db = Chime5()
chime5_it = chime5_db.get_iterator_by_names('train')

ex = sacred.Experiment('Chime5 train HMM GMM on array')

class MapFunKaldiInear2Array:
    def __init__(self, num_arrays=6, num_channels=4):
        self.num_arrays = num_arrays
        self.num_channels = num_channels
        id_list = [iterator[ids] for ids in iterator.keys()]
        self.map_dict = {self.get_inear_ids(ids, 'left'): self.get_ids(ids)
                         for ids in id_list}

    def get_inear_ids(self, id_dict, site):
        return id_dict['worn_microphone'][site]


    def get_ids(self, id_dict):
        return  (array[channel] for channel in range(self.num_channels)
                 for array in id_dict[keys.OBSERVATION].values()
                 if len(array)==4)

    def __call__(self, kaldi_id):
        return self.map_dict[kaldi_id]


def multiply_alignments(org_ali_dir, out_ali_dir, utt2split_dict,
                        id_map_fun: MapFunKaldiInear2Array(), num_jobs=16):
    assert not out_ali_dir.exists() or len(list(org_ali_dir.glob('*.gz'))) == 0,\
        f'The destination dir {out_ali_dir} already exists, please choose a' \
        f' different dir'
    num_alis = list(org_ali_dir.glob('ali.*.gz'))
    assert len(num_alis) == num_jobs, \
        f'number of alignment files {len(num_alis)} does not match,' \
        f'the number of jobs {num_jobs}'
    copytree(str(org_ali_dir), str(out_ali_dir), symlinks=True)
    [file.unlink() for file in out_ali_dir.glob('ali.*.gz')]

    def read_alignments(file):
        ali_decompressed = gzip.decompress(file.read_bytes())
        split_dict = {str(idx): list() for idx in range(1, num_jobs + 1)}
        for ex in ali_decompressed.decode().split('\n'):
            if ex == '':
                break
            utt, utt_ali = ex.split(' ', maxsplit=1)
            new_utts = id_map_fun(utt)
            for utt in new_utts:
                if utt in utt2split_dict:
                    split_dict[utt2split_dict[utt]].append(f'{utt} {utt_ali}')
        return split_dict

    global_split_dict = {str(idx): list() for idx in range(1, num_jobs + 1)}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for split_dict in tqdm.tqdm(
                ex.map(
                    read_alignments,
                    org_ali_dir.glob('ali*.gz'),
                ),
                total=num_jobs,
        ):
            for key in global_split_dict.keys():
                global_split_dict[key] += split_dict[key]
    global_split_dict = {key: set(value)
                         for key, value in global_split_dict.items()}
    def write_alignments(split):
        (out_ali_dir / f'ali.{split}.gz').write_bytes(gzip.compress(
            '\n'.join(global_split_dict[split]).encode()))

    with ThreadPoolExecutor(os.cpu_count()) as ex:
            list(tqdm.tqdm(
                    ex.map(
                        write_alignments,
                        global_split_dict.keys(),
                    ),
                    total=num_jobs
            ))


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

def calculate_mfccs(base_dir, train_set, num_jobs=20):
    if not (base_dir / 'data' / train_set / 'feats.scp').exists():
        run_process([
            f'{base_dir}/steps/make_mfcc.sh', '--nj', str(num_jobs), '--cmd', 'run.pl',
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

def calculate_alignments(base_dir, hmm_dir, train_set,
                         org_dir, ali_dir, num_jobs=16):
    if not (base_dir / 'data' / train_set).exists():
        copytree(str(org_dir / 'data' / train_set),
                 str(base_dir / 'data' / train_set))

    calculate_mfccs(base_dir, train_set)

    run_process([
        f'{base_dir}/steps/align_si.sh', '--nj', str(num_jobs), '--cmd', 'run.pl',
        f'data/{train_set}', 'data/lang',
        f'exp/{hmm_dir}', str(ali_dir)],
        cwd=str(base_dir), stdout=None, stderr=None
    )


def train_hmm_array(base_dir: Path, train_set='train_uall',
                    dev_set='dev_beamformit_ref',  org_hmm='tri3_worn',
                    org_train_set='train_worn', org_dir=ORG_DIR,
                    num_arrays=6, num_channels=4, num_jobs=16):
    get_files(base_dir, train_set, dev_set, org_dir)
    check_dest_dir(base_dir)
    train_dir = base_dir / 'data' / train_set
    out_ali_dir = base_dir / 'exp' / f'{org_hmm}_{train_set}_ali'
    if not (train_dir / 'feats.scp').exists():
        print('calculating mfcc features')
        calculate_mfccs(base_dir, train_set)
    if not (train_dir / f'split{num_jobs}').exists():
        run_process([
            f'{base_dir}/utils/split_data.sh.sh',
            f'{train_dir}', 'num_jobs'],
            cwd=str(base_dir), stdout=None, stderr=None)
    if not out_ali_dir.exists() or len(list(out_ali_dir.glob('*.gz'))) == 0:
        org_ali_dir = base_dir / 'exp' / f'{org_hmm}_{org_train_set}_ali'
        print(f'{out_ali_dir} not found, creating it from {org_ali_dir}')
        if not org_ali_dir.exists():
            print(f'{org_ali_dir} not found, calculating alignments')
            calculate_alignments(base_dir, org_hmm, org_train_set,
                                 org_dir, org_ali_dir, num_jobs)
        assert len(list(org_ali_dir.glob('*.gz'))) != 0, f'No alignments found' \
                                                         f'in org_dir {org_ali_dir}'
        map_fun = MapFunKaldiInear2Array(num_arrays, num_channels)
        utt2split_dict = get_utt2split_dict(train_dir, num_jobs)
        multiply_alignments(org_ali_dir, out_ali_dir, utt2split_dict, map_fun)
    else:
        alis = list(out_ali_dir.glob('ali.*.gz'))
        assert len(alis) == num_jobs, \
            f'number of alignment files {len(alis)} does not match,' \
            f'the number of jobs {num_jobs}'
    # fix_segment_file(base_dir / 'data' / train_set)
    run_process([
        f'{base_dir}/get_array_model.bash',
        '--train_set', f'{train_set}',
        '--dev_sets', f'{dev_set}',
        '--org_hmm', f'{org_hmm}'],
        cwd=str(base_dir),
        stdout=None, stderr=None
    )

def get_utt2split_dict(train_dir, num_jobs=16):
    split_dir = train_dir / f'split{num_jobs}'
    assert split_dir.exists()
    utt2split_dict = dict()
    for dirs in split_dir.glob('*'):
        ids = [lines.split(' ', maxsplit=1)[0]
               for lines in (dirs / 'text').open().readlines()]
        utt2split_dict.update({_id: dirs.name for _id in ids})
    return utt2split_dict



def fix_segment_file(train_set):
    print('fix the segment file ')
    def invert(iterator, key_list):
        def get_value(example):
            out = example
            for key in key_list:
                out = out[key]
            return out

        values = [get_value(iterator[key]) for key in iterator.keys()]
        return {out: key for out_dict, key in zip(values, iterator.keys())
                for out_list in out_dict.values() for out in out_list}

    segment_file = train_set / 'segments'
    assert segment_file.exists(), f'No segments file in {train_set}, please' \
                                  f' make sure the training directory exists'
    segment_zw = segment_file.parents[0] / ('zw_' + segment_file.name)
    segment_file.rename(segment_zw)

    it_mapped = chime5_it.map(adjust_start_end)
    mapping_inv = invert(iterator, ['observation'])
    segment_list = []
    for line in segment_zw.open().readlines():
        example_id, info, start, end = line.split(' ')
        array = info.split('.')[0].split('_')[1]
        channel = int(info[-1]) - 1
        nt_id = mapping_inv[example_id]
        nt_example = it_mapped[nt_id]
        start_sample = nt_example['start']['observation'][array][channel]
        end_sample = nt_example['end']['observation'][array][channel]
        start_new = '%013.7f' % (float(start_sample) / 16000)
        end_new = '%013.7f' % (float(end_sample) / 16000)
        segment_list.append(
            ' '.join([example_id, info, start_new, end_new + '\n']))
    segment_file.open('w').writelines(segment_list)
    segment_zw.unlink()


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