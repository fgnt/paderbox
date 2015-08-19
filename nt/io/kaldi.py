import subprocess
import tempfile
import os

import numpy as np

from nt.utils import mkdir_p


def import_feature_data(ark,
                        is_zipped=False):
    """ Read data from a kaldi ark file.

    Since the binary form is not documented and may change in future release, a kaldi tool (copy-feats) is used to
    first create a ark file in text mode. This file is then parsed and deleted afterwards.

    :param ark: The ark file to read
    :param copy_feats: The location of the kaldi tool `copy-feats`
    :return: A dictionary with the file ids as keys and their data as values
    """

    copy_cmd = '/net/ssd/software/kaldi/src/featbin/copy-feats'

    data = dict()
    with tempfile.NamedTemporaryFile() as tmp_ark:
        if is_zipped:
            src_param = 'ark:gunzip -c {ark}|'.format(ark=ark)
        else:
            src_param = 'ark:{ark}'.format(ark=ark)
        dest_param = 'ark,t:{tmp_ark}'.format(tmp_ark=tmp_ark.name)
        copy_process = subprocess.Popen([copy_cmd, src_param, dest_param],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

        out, err = copy_process.communicate()
        out = out.decode('ascii')
        err = err.decode('ascii')
        assert out == '', 'Copy-feats output is {}, expected nothing'.format(
            out)
        pos = err.find('Copied') + 1 + len('Copied')
        matrix_number = int(err[pos:].split()[0])
        with open(tmp_ark.name) as fid:
            raw_data = fid.read()
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


def import_alignment_data(ark,
                          is_zipped=True):
    """ Read data from a kaldi ark file.

    Since the binary form is not documented and may change in future release, a kaldi tool (copy-feats) is used to
    first create a ark file in text mode. This file is then parsed and deleted afterwards.

    :param ark: The ark file to read
    :param copy_feats: The location of the kaldi tool `copy-feats`
    :return: A dictionary with the file ids as keys and their data as values
    """

    copy_cmd = '/net/ssd/software/kaldi/src/bin/copy-int-vector'

    data = dict()
    with tempfile.NamedTemporaryFile() as tmp_ark:
        if is_zipped:
            src_param = 'ark:gunzip -c {ark}|'.format(ark=ark)
        else:
            src_param = 'ark:{ark}'.format(ark=ark)
        dest_param = 'ark,t:{tmp_ark}'.format(tmp_ark=tmp_ark.name)
        copy_process = subprocess.Popen([copy_cmd, src_param, dest_param],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

        out, err = copy_process.communicate()
        out = out.decode('ascii')
        err = err.decode('ascii')
        assert out == '', 'Copy-feats output is {}, expected nothing'.format(
            out)
        pos = err.find('Copied') + 1 + len('Copied')
        matrix_number = int(err[pos:].split()[0])
        with open(tmp_ark.name) as fid:
            raw_data = fid.readlines()
        for line in raw_data:
            split = line.split()
            data[split[0]] = np.asarray(split[1:], dtype=np.int32)
        assert len(data) == matrix_number, \
            'Copy-int-vector copied {num_matrix} vectors, but we read {num_data}'. \
                format(num_matrix=matrix_number, num_data=len(data))
    return data


def export_ark_data(data, ark_filename,
                    copy_feats='/net/ssd/software/kaldi/src/featbin/copy-feats'):
    """ Exports data to kaldi ark files.

    The data is first exported as a text ark file and then copied using kaldi copy-feats to a binary ark file and its
    final destination

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
    for ark in unique_arks:
        data_dict.update(import_feature_data(ark))
    assert len(set(list(data_dict.keys()) + utt_ids)) == len(data_dict), \
        'The scp contains utterances not present in the arks or vice versa'
    return data_dict


def import_alignment(ali_dir):
    data_dict = dict()
    for file in os.listdir(ali_dir):
        if file.startswith('ali'):
            ali_file = os.path.join(ali_dir, file)
            data_dict.update(import_alignment_data(ali_file))
    return data_dict


def import_features_and_alignment(feat_scp, ali_dir):
    features = import_feat_scp(feat_scp)
    alignments = import_alignment(ali_dir)
    common_length = len(set(list(features.keys()) + list(alignments.keys())))
    assert common_length == len(features) and common_length == len(alignments), \
        'Read {} features but {} alignments'.format(len(features),
                                                    len(alignments))
    return features, alignments
