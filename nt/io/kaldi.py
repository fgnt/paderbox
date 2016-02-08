import os
import subprocess
import warnings
import io

import numpy as np
import tqdm

from nt.utils import mkdir_p
from nt.utils.process_caller import run_processes

import struct
import tempfile

ENABLE_CACHE = True

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
                if ENABLE_CACHE:
                    self.cache[ark] = data
                return data

        return cached


feature_cache = FeatureCache()

KALDI_ROOT = os.environ.get('KALDI_ROOT', '/net/ssd/software/kaldi')

RAW_MFCC_CMD = KALDI_ROOT + '/src/featbin/' + \
    r"""compute-mfcc-feats --num-mel-bins={num_mel_bins} \
    --num-ceps={num_ceps} --low-freq={low_freq} --high-freq={high_freq} \
    scp,p:{wav_scp} ark,scp:{dst_ark},{dst_scp}"""

RAW_MFCC_DELTA_CMD = KALDI_ROOT + '/src/featbin/' + \
    r"""compute-mfcc-feats --num-mel-bins={num_mel_bins} \
    --num-ceps={num_ceps} --low-freq={low_freq} --high-freq={high_freq} \
    scp,p:{wav_scp} ark:- | add-deltas ark:- ark,scp:{dst_ark},{dst_scp}"""

RAW_FBANK_CMD = KALDI_ROOT + '/src/featbin/' + \
    r"""compute-fbank-feats --num-mel-bins={num_mel_bins} \
    --low-freq={low_freq} --high-freq={high_freq} \
    scp,p:{wav_scp} ark,scp:{dst_ark},{dst_scp}"""

RAW_FBANK_DELTA_CMD = KALDI_ROOT + '/src/featbin/' + \
    r"""compute-fbank-feats --num-mel-bins={num_mel_bins} \
    --low-freq={low_freq} --high-freq={high_freq} \
    scp,p:{wav_scp} ark:- | add-deltas ark:- ark,scp:{dst_ark},{dst_scp}"""


def make_mfcc_features(wav_scp, dst_dir, num_mel_bins, num_ceps, low_freq=20,
                       high_freq=-400, num_jobs=20, add_deltas=True):
    wav_scp = read_scp_file(wav_scp)
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
                if add_deltas:
                    cmd = RAW_MFCC_DELTA_CMD
                else:
                    cmd = RAW_MFCC_CMD
                cmds.append(cmd.format(
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


def make_fbank_features(wav_scp, dst_dir, num_mel_bins, low_freq=20,
                       high_freq=-400, num_jobs=20, add_deltas=True):
    wav_scp = read_scp_file(wav_scp)
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
                if add_deltas:
                    cmd = RAW_FBANK_DELTA_CMD
                else:
                    cmd = RAW_FBANK_CMD
                cmds.append(cmd.format(
                    num_mel_bins=num_mel_bins,
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


def import_feature_data(ark_descriptor):
    split = ark_descriptor.split(':')
    if len(split) == 1:
        ark = split[0]
        pos = 0
    elif len(split) == 2:
        ark = split[0]
        pos = int(split[1]) - 1
    else:
        raise ValueError('Cannot handle ark descriptor {}. Expected a format '
                         '"ark_file" or "ark_file:pos".'.format(ark_descriptor))
    if pos == 0:
        data = dict()
        with open(ark, 'rb') as ark_read_buffer:
            while True:
                utt_id = ''
                next_char = ark_read_buffer.read(1)
                if next_char == ''.encode():
                    break
                else:
                    c = struct.unpack('<s', next_char)[0]
                while c != ' '.encode():
                    utt_id += c.decode('utf8')
                    c = struct.unpack('<s', ark_read_buffer.read(1))[0]
                pos = ark_read_buffer.tell() - 1
                header = struct.unpack('<xccccbibi', ark_read_buffer.read(15))
                data[utt_id] = read_ark_mat(ark, pos)
                rows, colums = header[-3], header[-1]
                ark_read_buffer.seek(ark_read_buffer.tell() + (rows*colums*4))
        return data
    else:
        return read_ark_mat(ark, pos)


def read_ark_mat(ark, pos):
    with open(ark, 'rb') as ark_read_buffer:
        ark_read_buffer.seek(pos, 0)
        header = struct.unpack('<cxcccc', ark_read_buffer.read(6))
        if header[1] != "B".encode():
            raise ValueError("Input .ark file {} is not binary".format(ark))

        if header[2] == b'C':
            raise ValueError("Input .ark file {} is compress. You have to "
                             "decompress it first using copy-feats.".format(ark))

        m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        tmp_mat = np.frombuffer(
                ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
        utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

    return utt_mat


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
    feat_scp = read_scp_file(feat_scp)
    utt_ids = list(feat_scp.keys())
    data_dict = dict()
    print('Reading {} features'.format(len(utt_ids)))
    for utt_id in tqdm.tqdm(utt_ids):
        data_dict[utt_id] = import_feature_data(feat_scp[utt_id])
    return data_dict


def read_scp_file(scp_file):
    if isinstance(scp_file, dict):
        return scp_file
    scp_feats = dict()
    with open(scp_file) as fid:
        lines = fid.readlines()
    for line in lines:
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
