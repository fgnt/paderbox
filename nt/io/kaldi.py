import os
import subprocess
import warnings

import numpy as np
import tqdm

from nt.utils import mkdir_p
from nt.utils.process_caller import run_processes

import struct
import tempfile


class FeatureCache():
    def __init__(self):
        self.cache = dict()

    def __call__(self, fcn):
        def cached(*args, **kwargs):
            ark = args[0]
            if ark in self.cache:
                return self.cache[ark]
            else:
                data = fcn(*args, **kwargs)
                self.cache[ark] = data
                return data

        return cached


feature_cache = FeatureCache()

KALDI_ROOT = os.environ.get('KALDI_ROOT', '/net/ssd/software/kaldi')

RAW_MFCC_CMF = KALDI_ROOT + '/src/featbin/' + \
    r"""compute-mfcc-feats --num-mel-bins={num_mel_bins} \
    --num-ceps={num_ceps} --low-freq={low_freq} --high-freq={high_freq} \
    scp,p:{wav_scp} ark,scp:{dst_ark},{dst_scp}"""


def make_mfcc_features(wav_scp, dst_dir, num_mel_bins, num_ceps, low_freq=20,
                       high_freq=-400, num_jobs=20):
    wav_scp = read_scp_file(wav_scp, clean=False)
    split_mod = (len(wav_scp) // num_jobs) + 1
    print('Splitting jobs every {} ark'.format(split_mod))
    scp_idx = 0
    mkdir_p(dst_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmds = list()
        cur_scp = dict()
        for idx, (utt_id, ark) in enumerate(wav_scp.items()):
            cur_scp[utt_id] = ark
            if (not ((idx+1) % split_mod)) or (idx == (len(wav_scp) - 1)):
                with open(os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                          'w') as fid:
                    for _utt_id, _ark in cur_scp.items():
                        fid.write('{} {}\n'.format(_utt_id, _ark))
                cmds.append(RAW_MFCC_CMF.format(
                    num_mel_bins=num_mel_bins, num_ceps=num_ceps,
                    low_freq=low_freq, high_freq=high_freq,
                    wav_scp=os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                    dst_ark=os.path.join(dst_dir, '{}.ark'.format(scp_idx)),
                    dst_scp=os.path.join(dst_dir, '{}.scp'.format(scp_idx)),
                ))
                cur_scp = dict()
                scp_idx += 1
        print('Starting the feature extraction')
        run_processes(cmds, sleep_time=5)
        with open(os.path.join(dst_dir, 'feats.scp'), 'w') as feat_fid:
            for f in os.listdir(dst_dir):
                if f.endswith('.scp'):
                    with open(os.path.join(dst_dir, f), 'r') as fid:
                        feat_fid.writelines(fid.readlines())
        feat_scp = read_scp_file(os.path.join(dst_dir, 'feats.scp'))
        if len(feat_scp) != len(wav_scp):
            missing = np.setdiff1d(np.unique(list(wav_scp.keys())),
                                   np.unique(list(feat_scp.keys())))
            raise ValueError(
                    'Mismatch between number of wav files and number '
                    'of feature files. Missing the utterances {}'.
                        format(missing))
        print('Finished successfully')


@feature_cache
def import_feature_data(ark, is_zipped=False):
    """ Read data from a kaldi ark file.

    Since the binary form is not documented and may change in future release, a
    kaldi tool (copy-feats) is used to first create a ark file in text mode.
    This output is then parsed

    :param ark: The ark file to read
    :return: A dictionary with the file ids as keys and their data as values
    """

    copy_cmd = '/net/ssd/software/kaldi/src/featbin/copy-feats'

    data = dict()
    if is_zipped:
        src_param = 'ark,p:gunzip -c {ark}|'.format(ark=ark)
    else:
        src_param = 'ark,p:{ark}'.format(ark=ark)
    dest_param = 'ark,t:-'
    copy_process = subprocess.Popen([copy_cmd, src_param, dest_param],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

    out, err = copy_process.communicate()
    if copy_process.returncode != 0:
        raise ValueError("Returncode of copy-feats was != 0. Stderr "
                         "output is: {}".format(err))
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    pos = err.find('Copied') + 1 + len('Copied')
    matrix_number = int(err[pos:].split()[0])
    raw_data = out
    utterances = raw_data.split(']')
    for utt in utterances:
        if len(utt.split('[')) > 1:
            utt_name = utt.split('[')[0].split(' ')[0].replace('\n', '')
            utt_data_raw = utt.split('[')[1].split('\n')[1:]
            feature_len = len(utt_data_raw[0].lstrip().rstrip().split(' '))
            np_data = np.zeros((len(utt_data_raw), feature_len),
                               dtype=np.float32)
            for idx, row in enumerate(utt_data_raw):
                np_data[idx, :] = row.lstrip().rstrip().split(' ')
            data[utt_name] = np_data
    assert len(data) == matrix_number, \
        'Copy-feats copied {num_matrix} matrices, but we read {num_data}'. \
            format(num_matrix=matrix_number, num_data=len(data))
    return data


def import_alignment_data(ark, model_file, is_zipped=True):
    """ Read data from a kaldi ark file.

    Since the binary form is not documented and may change in future release,
    a kaldi tool (ali-to-pdf) is used to first create a ark file in text mode.

    :param ark: The ark file to read
    :param model_file: Model file used to create the alignments. This is needed
        to extract the pdf ids
    :param copy_feats: The location of the kaldi tool `copy-feats`
    :return: A dictionary with the file ids as keys and their data as values
    """

    copy_cmd = '/net/ssd/software/kaldi/src/bin/ali-to-pdf'

    data = dict()
    if is_zipped:
        src_param = 'ark:gunzip -c {ark} |'.format(ark=ark)
    else:
        src_param = 'ark:{ark}'.format(ark=ark)
    dest_param = 'ark,t:-'
    copy_process = subprocess.Popen(
            [copy_cmd, model_file, src_param, dest_param],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    out, err = copy_process.communicate()
    if copy_process.returncode != 0:
        raise ValueError("Returncode of ali-to-pdf was != 0. Stderr "
                         "output is:\n{}".format(err))
    out = out.decode('utf-8')
    err = err.decode('utf-8')
    pos = err.find('Converted') + 1 + len('Converted')
    matrix_number = int(err[pos:].split()[0])
    for line in out.split('\n'):
        split = line.split()
        if len(split) > 0:
            utt_id = split[0]
            ali = np.asarray(split[1:], dtype=np.int32)
            data[utt_id] = ali
    assert len(data) == matrix_number, \
        'ali-to-pdf converted {num_matrix} alignments, ' \
        'but we read {num_data}'. \
            format(num_matrix=matrix_number, num_data=len(data))
    return data


class ArkWriter():
    def __init__(self, ark_filename):
        self.ark_path = ark_filename

    def __enter__(self):
        self.ark_file_write = open(self.ark_path, "wb")
        return self

    def write_array(self, utt_id, array):
        utt_mat = np.asarray(array, dtype=np.float32)
        rows, cols = utt_mat.shape
        self.ark_file_write.write(
                struct.pack('<%ds'%(len(utt_id)), utt_id.encode()))
        self.ark_file_write.write(
                struct.pack('<cxcccc',
                            ' '.encode(),
                            'B'.encode(),
                            'F'.encode(),
                            'M'.encode(),
                            ' '.encode()))
        self.ark_file_write.write(struct.pack('<bi', 4, rows))
        self.ark_file_write.write(struct.pack('<bi', 4, cols))
        self.ark_file_write.write(utt_mat)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ark_file_write.close()


def import_feat_scp(feat_scp):
    if not isinstance(feat_scp, dict):
        feat_scp = read_scp_file(feat_scp)
    utt_ids = list(feat_scp.keys())
    arks = list(feat_scp.values())
    unique_arks = sorted(set(arks))
    data_dict = dict()
    print('Reading {} feature arks'.format(len(unique_arks)))
    for ark in tqdm.tqdm(unique_arks):
        data_dict.update(import_feature_data(ark))
    assert len(set(list(data_dict.keys()) + utt_ids)) == len(data_dict), \
        'The scp contains utterances not present in the arks'
    data_dict = {key: data_dict[key] for key in utt_ids}
    return data_dict


def read_scp_file(scp_file, clean=True):
    scp_feats = dict()
    with open(scp_file) as fid:
        lines = fid.readlines()
    for line in lines:
        if clean:
            scp_feats[line.split()[0]] = line.split()[1].split(':')[0]
        else:
            scp_feats[line.split()[0]] = ' '.join(line.split()[1:])
    return scp_feats


def import_alignment(ali_dir, model_file):
    """ Imports an alignments (pdf-ids)

    :param ali_dir: Directory containing the ali.* files
    :param model_file: Model used to create the alignments
    :return: Dict with utterances as key and alignments as value
    """
    data_dict = dict()
    print('Importing alignments')
    for file in os.listdir(ali_dir):
        if file.startswith('ali'):
            ali_file = os.path.join(ali_dir, file)
            data_dict.update(import_alignment_data(ali_file, model_file))
    return data_dict


def import_features_and_alignment(feat_scp, ali_dir, model_file):
    """ Import features and alignments given a scp file and a ali directory

    This is basically a wrapper around the other functions

    .. note:: This wrapper checks if the utterance ids match

    :param feat_scp: Scp file describing the features
    :param ali_dir: Directory with alignments
    :param model_file: Model used to create alignments
    :return: Tuple of (dict, dict) where the first contains the features and
        the second the corresponding alignments
    """
    features = import_feat_scp(feat_scp)
    alignments = import_alignment(ali_dir, model_file)
    common_keys = set(features.keys()).intersection(set(alignments.keys()))
    common_length = len(common_keys)
    if common_length != len(features) or common_length != len(alignments):
        warnings.warn('Found features for {} and alignments for {} utterances.'
                      ' Returning common set.'.format(len(features),
                                                      len(alignments)))
    features_common = dict()
    alignments_common = dict()
    for utt_id in common_keys:
        features_common[utt_id] = features[utt_id]
        alignments_common[utt_id] = alignments[utt_id]
        assert features[utt_id].shape[0] == alignments[utt_id].shape[0], \
            'There are {} features for utterance {} but {} alignments'.format(
                    features[utt_id].shape[0], utt_id,
                    alignments[utt_id].shape[0]
            )
    return features_common, alignments_common
