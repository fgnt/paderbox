"""
Example call on nt:
Automatically takes all available cores:
python -m chime5.scripts.kaldi_array -F ~/sacred/chime5/arrayBSS/54/kaldi/inear with inear audio_dir=../../audio/dev

Example call on pc2:
ccsalloc --group=hpc-prf-nt1 --res=rset=64:vmem=2G:mem=2G:ncpus=1 -t 6h python -m chime5.scripts.kaldi_array -F ~/sacred/chime5/arrayBSS/54/kaldi/inear with inear audio_dir=../../audio/dev num_jobs=64

"""

import os

from pathlib import Path
from shutil import copytree, copyfile

import sacred

from chime5.scripts.kaldi import ORG_DIR
from nt.database.chime5 import Chime5
from nt.utils.process_caller import run_process
from nt.io.data_dir import database_jsons, kaldi_root
from nt.io import symlink, mkdir_p

ex = sacred.Experiment('Kaldi array')

db = Chime5(database_jsons / 'chime5_orig.json')

NEEDED_FILES = ['cmd.sh', 'path.sh', 'get_model.bash']
NEEDED_DIRS = ['data/lang', 'data/local', 'data/srilm', 'conf', 'local']



@ex.capture
def calculate_mfccs(base_dir, dataset, num_jobs=20, config='mfcc.conf',
                    recalc=False, kaldi_cmd='run.pl'):
    '''

    :param base_dir: kaldi egs directory with steps and utils dir
    :param dataset: name of folder in data
    :param num_jobs: number of parallel jobs
    :param config: mfcc config
    :param recalc: recalc feats if already calculated
    :return:
    '''
    if isinstance(dataset, str):
        dataset = base_dir / 'data' / dataset
    assert dataset.exists(), dataset
    if not (dataset / 'feats.scp').exists() or recalc:
        run_process([
            'steps/make_mfcc.sh', '--nj', str(num_jobs),
            '--mfcc-config', f'{base_dir}/conf/{config}',
            '--cmd', f'{kaldi_cmd}', f'{dataset}',
            f'{dataset}/make_mfcc', f'{dataset}/mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )

    if not (dataset / 'cmvn.scp').exists() or recalc:
        run_process([
            f'steps/compute_cmvn_stats.sh',
            f'{dataset}', f'{dataset}/make_mfcc', f'{dataset}/mfcc'],
            cwd=str(base_dir), stdout=None, stderr=None
        )
    run_process([
        f'utils/fix_data_dir.sh', f'{dataset}'],
        cwd=str(base_dir), stdout=None, stderr=None
    )


@ex.capture
def calculate_ivectors(ivector_dir, dest_dir, org_dir, train_affix, dataset_dir,
                       extractor_dir, num_jobs=8, kaldi_cmd='run.pl'):
    '''
    
    :param ivector_dir: ivector directory may be a string, bool or Path
    :param base_dir: kaldi egs directory with steps and utils dir
    :param train_affix: dataset specifier (dataset name without train or dev)
    :param dataset_dir: kaldi dataset dir
    :param extractor_dir: directory of the ivector extractor (may be None)
    :param num_jobs: number of parallel jobs
    :return: 
    '''
    if isinstance(ivector_dir, str):
        ivector_dir = dest_dir / 'exp' / ('nnet3_' + train_affix) / ivector_dir
    elif isinstance(ivector_dir, bool):
        ivector_dir = dest_dir / 'exp' / ('nnet3_' + train_affix) / (
            'ivectors_' + dataset_dir.name)
    elif isinstance(ivector_dir, Path):
        ivector_dir = ivector_dir
    else:
        raise ValueError(f'ivector_dir {ivector_dir} has to be either'
                         f' a Path, a string or bolean')
    if not ivector_dir.exists():
        if extractor_dir is None:
            extractor_dir = org_dir / f'exp/nnet3_{train_affix}/extractor'
        else:
            if isinstance(extractor_dir, str):
                extractor_dir = org_dir / f'exp/{extractor_dir}'
        assert extractor_dir.exists(), extractor_dir
        print(f'Directory {ivector_dir} not found, estimating ivectors')
        run_process([
            'steps/online/nnet2/extract_ivectors_online.sh',
            '--cmd', f'{kaldi_cmd}', '--nj', f'{num_jobs}', f'{dataset_dir}',
            f'{extractor_dir}', str(ivector_dir)],
            cwd=str(dest_dir),
            stdout=None, stderr=None
        )
    return ivector_dir

@ex.capture
def copy_ref_dir(dev_dir, ref_dir, audio_dir, allow_missing_files=False):
    mapping = db.kaldi_id_mapper()
    required_files = ['utt2spk', 'text']
    with (ref_dir / 'text').open() as file:
        text = file.readlines()
    ref_ids = [line.split(' ', maxsplit=1)[0].strip() for line in text]
    mkdir_p(dev_dir)
    for files in required_files:
        copyfile(str(ref_dir / files), str(dev_dir / files))
    ids = {
    wav_file: mapping.get_array_ids_from_nt_id(wav_file.name.split('.')[0],
                                               channels='ENH')
    for wav_file in audio_dir.glob('*')}
    used_ids = {kaldi_id: wav_file for wav_file, kaldi_ids in ids.items()
                for kaldi_id in kaldi_ids if kaldi_id in ref_ids}
    assert len(used_ids) > 0, used_ids
    if len(used_ids) < len(ids):
        print(f'Not all files in {audio_dir} were used, '
              f'{len(ids)-len(used_ids)} ids are not used in kaldi')
    elif len(used_ids) < len(ref_ids):
        if not allow_missing_files:
            raise ValueError(
                f'{len(ref_ids)-len(used_ids)} files are missing in {audio_dir}.'
                f' We found only {len(used_ids)} files but expect {len(ref_ids)}'
                f' files')
        print(f'{len(ref_ids)-len(used_ids)} files are missing in {audio_dir}.'
              f' We found only {len(used_ids)} files but expect {len(ref_ids)}'
              f' files. Still continuing to decode the remaining files')
        ref_ids = [_id for _id in used_ids.keys()]
        ref_ids.sort()
        for files in ['utt2spk', 'text']:
            with (dev_dir / files).open() as fd:
                lines = fd.readlines()
                lines = [line for line in lines
                         if line.split(' ')[0] in used_ids]
            (dev_dir / files).unlink()
            with (dev_dir / files).open('w') as fd:
                fd.writelines(lines)

    wavs = [' '.join([kaldi_id, str(used_ids[kaldi_id])]) + '\n'
            for kaldi_id in ref_ids]
    with (dev_dir / 'wav.scp').open('w') as file:
        file.writelines(wavs)


def get_dev_dir(base_dir: Path, org_dir: Path, enh='bss_beam',
                hires=True, ref_dir='dev_beamformit_ref',
                audio_dir=None, num_jobs=8):
    if isinstance(enh, Path):
        dev_dir = enh
    elif 'hires' in enh and hires:
        dev_dir = base_dir / 'data' / f'dev_{enh}'
    elif hires:
        dev_dir = base_dir / 'data' / f'dev_{enh}_hires'
    else:
        dev_dir = base_dir / 'data' / f'dev_{enh}'
    config = 'mfcc_hires.conf' if hires else 'mfcc.conf'
    if not dev_dir.exists():
        print(f'Directory {dev_dir} not found creating data directory')
        if isinstance(ref_dir, str):
            ref_dir = org_dir / 'data' / ref_dir
        assert ref_dir.exists(), ref_dir
        if audio_dir is None:
            copytree(str(ref_dir), str(dev_dir))
        else:
            assert audio_dir.exists(), audio_dir
            copy_ref_dir(dev_dir, ref_dir, audio_dir)
        run_process([
            f'{base_dir}/utils/fix_data_dir.sh', str(dev_dir)],
            cwd=str(base_dir), stdout=None, stderr=None
        )
        calculate_mfccs(base_dir, dev_dir, num_jobs=num_jobs,
                        config=config, recalc=True)
    return dev_dir


@ex.capture
def create_dest_dir(dest_dir, org_dir=ORG_DIR, kaldi_root=kaldi_root):
    dest_dir.mkdir(exist_ok=True)
    (dest_dir / 'data').mkdir(exist_ok=True)
    for file in NEEDED_FILES:
        symlink(org_dir / file, dest_dir / file)
    for dirs in NEEDED_DIRS:
        symlink(org_dir / dirs, dest_dir / dirs)
    for link in ['steps', 'utils']:
        if (org_dir / link).exists():
            linkto = os.readlink(org_dir / link)
        else:
            linkto = Path(kaldi_root) / f'egs/wsj/s5/{link}'
        symlink(linkto, dest_dir / link)

    if 'CCS_NODEFILE' in os.environ:
        CCS_NODEFILE = Path(os.environ['CCS_NODEFILE'])
        (dest_dir / '.queue').mkdir()
        (dest_dir / '.queue' / 'machines').write_text(CCS_NODEFILE.read_text())


@ex.capture
def decode(model_dir, dest_dir, org_dir, audio_dir: Path,
           ref_dir='dev_beamformit_ref', model_type='tdnn1a_sp',
           ivector_dir=False, extractor_dir=None,
           hires=True, enh='bss_beam', num_jobs=8, kaldi_cmd='run.pl'):
    '''

    :param model_dir: name of model or Path to model_dir
    :param dest_dir: kaldi egs dir for the decoding
    :param org_dir: kaldi egs dir from which information for decoding are gathered
    :param audio_dir: directory of audio files to decode (may be None)
    :param ref_dir: reference kaldi dataset directory or name for decode dataset
    :param ivector_dir: directory or name for the ivectors (may be None or False)
    :param extractor_dir: directory of the ivector extractor (maybe None)
    :param hires: flag for using high resolution mfcc features (True / False)
    :param enh: name of the enhancement method, used for creating dataset name
    :param num_jobs: number of parallel jobs
    :return:
    '''
    model_dir = org_dir / 'exp' / model_dir
    assert model_dir.exists(), f'{model_dir} does not exist'
    if not dest_dir == org_dir:
        create_dest_dir(dest_dir, org_dir)
    os.environ['PATH'] = f'{dest_dir}/utils:{os.environ["PATH"]}'
    train_affix = model_dir.name.split('_', maxsplit=1)[1]
    dataset_dir = get_dev_dir(dest_dir, org_dir, enh, hires, ref_dir,
                          audio_dir, num_jobs)
    decode_dir = dest_dir / f'exp/{model_dir.name}/{model_type}'
    if not decode_dir.exists():
        mkdir_p(decode_dir)
        [symlink(file, decode_dir / file.name)
         for file in (model_dir / model_type).glob('*') if file.is_file()]
        assert (decode_dir / 'final.mdl').exists(), (
            f'final.mdl not in decode_dir: {decode_dir},'
            f' maybe using worn org_dir: {org_dir}?'
        )
    (decode_dir / f'decode_{enh}').mkdir(exist_ok=False)
    if ivector_dir:
        ivector_dir = calculate_ivectors(ivector_dir, dest_dir, org_dir,
                                         train_affix, dataset_dir,
                                         extractor_dir, num_jobs)
        run_process([
            'steps/nnet3/decode.sh', '--acwt', '1.0',
            '--post-decode-acwt', '10.0',
            '--extra-left-context', '0', '--extra-right-context', '0',
            '--extra-left-context-initial', '0', '--extra-right-context-final',
            '0',
            '--frames-per-chunk', '140', '--nj', str(num_jobs), '--cmd',
            f'{kaldi_cmd}', '--online-ivector-dir',
            str(ivector_dir), f'{model_dir}/tree_sp/graph', str(dataset_dir),
            str(decode_dir / f'decode_{enh}')],
            cwd=str(dest_dir),
            stdout=None, stderr=None
        )
    else:
        run_process([
            'steps/nnet3/decode.sh', '--acwt', '1.0',
            '--post-decode-acwt', '10.0',
            '--extra-left-context', '0', '--extra-right-context', '0',
            '--extra-left-context-initial', '0', '--extra-right-context-final',
            '0',
            '--frames-per-chunk', '140', '--nj', str(num_jobs), '--cmd',
            f'"{kaldi_cmd} --mem 4G"', '--num-threads', '4',
            f'{model_dir}/tree/graph', str(dataset_dir),
            str(decode_dir)],
            cwd=str(dest_dir),
            stdout=None, stderr=None
        )
    print((
              decode_dir / f'decode_{enh}' / 'scoring_kaldi' / 'best_wer'
          ).read_text())


def on_pc2():
    # len(Path(os.environ["CCS_NODEFILE"]).read_text().strip().split("\n"))
    if 'CCS_NODEFILE' in os.environ:
        return True
    else:
        return False


@ex.config
def default():
    """
    audio_dir: Dir to decode. The path can be absolute or relative to the sacred
        base folder. If None take the ref_dev_dir as audio_dir. Note when
        ref_dev_dir is a relative path, it is relative to org_dir and not sacred
        base folder.

    org_dir/model_dir: Folder of the trained model

    enh: Name of this experiment for the kaldi folders

    """
    org_dir = ORG_DIR
    model_dir = None
    audio_dir = None
    ivector_dir = False
    ref_dir = 'dev_beamformit_ref'
    enh = 'bss_beam'
    model_type = 'tdnn1a_sp'
    extractor_dir = None
    hires = True
    num_jobs = os.cpu_count()
    kaldi_root = None
    allow_missing_files = False

    if on_pc2():
        kaldi_cmd = 'ssh.pl'
    else:
        kaldi_cmd = 'run.pl'


@ex.named_config
def baseline():
    if on_pc2():
        kaldi_root = '/scratch/hpc-prf-nt1/fgnt/kaldi_2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb'
        org_dir = '/scratch/hpc-prf-nt1/fgnt/models_kaldi/chime5/baseline'
    else:
        kaldi_root = '/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/'
        org_dir = '/net/vol/jenkins/models_kaldi/chime5/baseline'
    model_dir = 'chain_train_worn_u100k'
    ivector_dir = True
    extractor_dir = None
    hires = True


@ex.named_config
def inear():
    if on_pc2():
        kaldi_root = '/scratch/hpc-prf-nt1/fgnt/kaldi_2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb'
        org_dir = '/scratch/hpc-prf-nt1/fgnt/models_kaldi/chime5/inear'
    else:
        kaldi_root = '/net/vol/jenkins/kaldi/2018-03-21_08-33-34_eba50e4420cfc536b68ca7144fac3cd29033adbb/'
        org_dir = '/net/vol/jenkins/models_kaldi/chime5/inear'
    model_dir = 'chain_worn_bss_stereo_cleaned'
    ivector_dir = True
    extractor_dir = None
    hires = True


def check_config_element(element):
    if element is not None and not isinstance(element, bool):
        element_path = element
        if Path(element_path).exists():
            element_path = Path(element_path)
    elif isinstance(element, bool):
        element_path = element
    else:
        element_path = None
    return element_path


@ex.automain
def run(_config, audio_dir, kaldi_root):
    if kaldi_root is None:
        kaldi_root = os.environ['KALDI_ROOT']
    assert Path(kaldi_root).exists(), kaldi_root
    os.environ['KALDI_ROOT'] = kaldi_root

    assert len(ex.current_run.observers) == 1, (
        'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
    )
    base_dir = Path(ex.current_run.observers[0].basedir)
    if audio_dir is not None:
        audio_dir = base_dir / audio_dir
        assert audio_dir.exists(), audio_dir
    if isinstance(_config['org_dir'], bool):
        org_dir = base_dir
    else:
        org_dir = Path(_config['org_dir'])
        assert org_dir.exists(), org_dir

    decode(model_dir=check_config_element(_config['model_dir']),
           dest_dir=base_dir,
           org_dir=org_dir,
           audio_dir=audio_dir,
           ref_dir=check_config_element(_config['ref_dir']),
           model_type=_config['model_type'],
           ivector_dir=check_config_element(_config['ivector_dir']),
           extractor_dir=check_config_element(_config['extractor_dir']),
           hires=_config['hires'],
           enh=check_config_element(_config['enh']),
           num_jobs=_config['num_jobs']
           )
