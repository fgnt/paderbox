import os
import subprocess
import tempfile
import warnings

import numpy as np
import tqdm

from nt.utils import mkdir_p


def import_feature_data(ark, is_zipped=False):
    """ Read data from a kaldi ark file.

    Since the binary form is not documented and may change in future release, a
    kaldi tool (copy-feats) is used to first create a ark file in text mode.
    This output is then parsed

    :param ark: The ark file to read
    :return: A dictionary with the file ids as keys and their data as values
    """
    print('Import ark {}'.format(ark))
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
    for utt in tqdm.tqdm(utterances):
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


def export_ark_data(data, ark_filename,
                    copy_feats='/net/ssd/software/kaldi/src/featbin/copy-feats'):
    """ Exports data to kaldi ark files.

    The data is first exported as a text ark file and then copied using kaldi
    copy-feats to a binary ark file and its final destination

    :param data: A dictionary with the data to export. The key is used for the id
    :param ark_filename: Destination of the ark file
    :param copy_feats: The location of the kaldi tool `copy-feats`
    :return:
    """
    mkdir_p(os.path.dirname(ark_filename))
    with tempfile.NamedTemporaryFile() as txt_ark:
        with open(txt_ark.name, 'w') as fid:
            for id, array in data.items():
                fid.write(id)
                fid.write(' [ ')
                for row in array:
                    for value in row:
                        fid.write('{0} '.format(value))
                    fid.write('\n')
                fid.write(']\n')
        src_param = 'ark,t:{txt_ark}'.format(txt_ark=txt_ark.name)
        dest_param = 'ark:{dst_ark}'.format(dst_ark=ark_filename)
        copy_process = subprocess.Popen([copy_feats, src_param, dest_param],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        out, err = copy_process.communicate()
        assert out == '', 'Copy-feats output is {}, expected nothing'.format(
                out)
        assert len(err.split(
                '\n')) == 3, 'Expected three lines on stderr, but got {}'.format(
                err)
        pos = err.split('\n')[1].find('Copied') + 1 + len('Copied')
        matrix_number = int(err.split('\n')[1][pos:].split()[0])
        assert matrix_number == len(data), \
            'Tried to export {num_data} matrices, but only {num_ark} got ' \
            'exported'.format(num_data=len(data), num_ark=matrix_number)


def import_feat_scp(feat_scp):
    with open(feat_scp) as fid:
        lines = fid.readlines()
    utt_ids = list()
    arks = list()
    for line in lines:
        utt_ids.append(line.split()[0])
        arks.append(line.split()[1].split(':')[0])
    unique_arks = set(arks)
    data_dict = dict()
    print('Reading feature arks')
    for ark in unique_arks:
        data_dict.update(import_feature_data(ark))
    assert len(set(list(data_dict.keys()) + utt_ids)) == len(data_dict), \
        'The scp contains utterances not present in the arks or vice versa'
    return data_dict


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
