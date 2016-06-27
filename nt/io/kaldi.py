import os
import struct
import subprocess
import tempfile
import warnings
from io import BytesIO

import numpy as np
import tqdm

from nt.io.audioread import audioread
from nt.io.audiowrite import audiowrite
from nt.io.data_dir import kaldi_root
from nt.utils import mkdir_p
from nt.utils.process_caller import run_processes

ENABLE_CACHE = True


def get_kaldi_env():
    env = os.environ.copy()
    env['PATH'] += ':{}/src/bin'.format(kaldi_root())
    env['PATH'] += ':{}/tools/openfst/bin'.format(kaldi_root())
    env['PATH'] += ':{}/src/fstbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/gmmbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/featbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/lm/'.format(kaldi_root())
    env['PATH'] += ':{}/src/sgmmbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/sgmm2bin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/fgmmbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/latbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/nnetbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/nnet2bin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/kwsbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/online2bin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/ivectorbin/'.format(kaldi_root())
    env['PATH'] += ':{}/src/lmbin/'.format(kaldi_root())
    if 'LD_LIBRARY_PATH' in env.keys():
        env["LD_LIBRARY_PATH"] += ":{}/tools/openfst/lib".format(kaldi_root())
    else:
        env["LD_LIBRARY_PATH"] = ":{}/tools/openfst/lib".format(kaldi_root())
    env['LC_ALL'] = 'C'
    env['OMP_NUM_THREADS'] = '1'
    return env


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

KALDI_ROOT = kaldi_root()

RAW_MFCC_CMD = KALDI_ROOT + '/src/featbin/' + \
               r"""compute-mfcc-feats --num-mel-bins={num_mel_bins} --use-energy={use_energy} \
               --num-ceps={num_ceps} --low-freq={low_freq} --high-freq={high_freq} \
               scp,p:{wav_scp} ark:- """

ADD_DELTA_CMD = ' add-deltas ark:- ark:- '

COMPUTE_NORMALIZE_CMD = 'compute-cmvn-stats scp:{feat_scp} scp:{cmvn_scp}'

NORMALIZE_CMD = 'apply-cmvn scp:{feat_scp} scp:{cmvn_scp} ark,scp:{dst_ark},{dst_scp}'

STORE_FEATS = ' copy-feats ark:- ark,scp:{dst_ark},{dst_scp}'

RAW_FBANK_CMD = KALDI_ROOT + '/src/featbin/' + \
                r"""compute-fbank-feats --num-mel-bins={num_mel_bins} \
                --low-freq={low_freq} --high-freq={high_freq} --use-energy={use_energy} \
                --use-log-fbank={use_log_fbank} --window-type={window_type} \
                scp,p:{wav_scp} ark:- """

RAW_FBANK_ARK_CMD = KALDI_ROOT + '/src/featbin/' + \
                r"""compute-fbank-feats --num-mel-bins={num_mel_bins} \
                --low-freq={low_freq} --high-freq={high_freq} --use-energy={use_energy} \
                --use-log-fbank={use_log_fbank} --window-type={window_type} \
                ark:- ark:- """


def _build_cmd(extractor, add_deltas=False, store_feats=False):
    cmds = [extractor]
    if add_deltas:
        cmds.append(ADD_DELTA_CMD)
    if store_feats:
        cmds.append(STORE_FEATS)
    return '|'.join(cmds)

def make_mfcc_features(wav_scp, dst_dir, num_mel_bins, num_ceps, low_freq=20,
                       high_freq=-400, num_jobs=20, add_deltas=True,
                       use_energy=False):
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
            if (not ((idx + 1) % split_mod)) or (idx == (len(wav_scp) - 1)):
                with open(os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                          'w') as fid:
                    for _utt_id, _ark in cur_scp.items():
                        fid.write('{} {}\n'.format(_utt_id, _ark))

                cmd = _build_cmd(RAW_MFCC_CMD, add_deltas=add_deltas,
                                 store_feats=True)

                cmds.append(cmd.format(
                    num_mel_bins=num_mel_bins, num_ceps=num_ceps,
                    low_freq=low_freq, high_freq=high_freq,
                    use_energy=use_energy,
                    wav_scp=os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                    dst_ark=os.path.join(dst_dir, '{}.ark'.format(scp_idx)),
                    dst_scp=os.path.join(dst_dir, '{}.scp'.format(scp_idx)),
                ))
                cur_scp = dict()
                scp_idx += 1
        print('Starting the feature extraction')
        run_processes(cmds, sleep_time=5, environment=get_kaldi_env())
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
                        high_freq=-400, num_jobs=20, add_deltas=True,
                        use_energy=False,
                        use_log_fbank=True, window_type="povey", verbose=False):
    wav_scp = read_scp_file(wav_scp)
    split_mod = (len(wav_scp) // num_jobs) + 1
    if verbose:
        print('Splitting jobs every {} ark'.format(split_mod))
    scp_idx = 0
    mkdir_p(dst_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmds = list()
        cur_scp = dict()
        for idx, (utt_id, ark) in enumerate(wav_scp.items()):
            cur_scp[utt_id] = ark
            if (not ((idx + 1) % split_mod)) or (idx == (len(wav_scp) - 1)):
                with open(os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                          'w') as fid:
                    for _utt_id, _ark in cur_scp.items():
                        fid.write('{} {}\n'.format(_utt_id, _ark))

                cmd = _build_cmd(RAW_FBANK_CMD, add_deltas=add_deltas,
                                 store_feats=True)

                cmds.append(cmd.format(
                    num_mel_bins=num_mel_bins, use_energy=use_energy,
                    low_freq=low_freq, high_freq=high_freq,
                    use_log_fbank=use_log_fbank, window_type=window_type,
                    wav_scp=os.path.join(tmp_dir, '{}.scp'.format(scp_idx)),
                    dst_ark=os.path.join(dst_dir, '{}.ark'.format(scp_idx)),
                    dst_scp=os.path.join(dst_dir, '{}.scp'.format(scp_idx)),
                ))
                cur_scp = dict()
                scp_idx += 1
        if verbose:
            print('Starting the feature extraction')
        run_processes(cmds, sleep_time=5, environment=get_kaldi_env())
        with open(os.path.join(dst_dir, 'feats.scp'), 'w') as feat_fid:
            for f in os.listdir(dst_dir):
                if f.endswith('.scp'):
                    with open(os.path.join(dst_dir, f), 'r') as fid:
                        feat_fid.writelines(fid.readlines())
        feat_scp = read_scp_file(os.path.join(dst_dir, 'feats.scp'))
        if len(feat_scp) != len(wav_scp):
            missing = np.setdiff1d(np.unique(list(wav_scp.keys())),
                                   np.unique(list(feat_scp.keys())))
            warnings.warn(
                'Mismatch between number of wav files and number '
                'of feature files. Missing the utterances {}'.format(missing))
        if verbose:
            print('Finished successfully')


def write_utt2spk(dst_dir, utt2spk):
    with open(os.path.join(dst_dir, 'utt2spk'), 'w') as fid:
        for utt, spk in utt2spk.items():
            fid.write('{} {}\n'.format(utt, spk))


def write_spk2utt(dst_dir, utt2spk):
    with open(os.path.join(dst_dir, 'spk2utt'), 'w') as fid:
        for spk in set(utt2spk.values()):
            fid.write('{} '.format(spk))
            fid.write(' '.join([utt for utt in utt2spk if utt2spk[utt] == spk]))
            fid.write('\n')


def compute_mean_and_var_stats(feat_scp, dst_dir, utt2spk=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        mkdir_p(dst_dir)
        if utt2spk is not None:
            write_spk2utt(tmp_dir, utt2spk)
            spk2utt = '--spk2utt=ark:{} '.format(os.path.join(tmp_dir, 'spk2utt'))
        else:
            spk2utt = ''
        cmd = 'compute-cmvn-stats {2}scp:{0} ark:{1}/cmvn.ark'.format(
            feat_scp, dst_dir, spk2utt
        )
        run_processes(cmd, environment=get_kaldi_env())


def apply_mean_and_var_stats(feat_scp, cmvn_ark, utt2spk=None, norm_var=False):
    with tempfile.TemporaryDirectory() as tmp_dir:
        if utt2spk is not None:
            write_utt2spk(tmp_dir, utt2spk)
            args = '--utt2spk=ark:{} '.format(os.path.join(tmp_dir, 'utt2spk'))
        else:
            args = ''
        if norm_var:
            args += '--norm-vars '
        cmd = 'apply-cmvn {args}ark:{cmvn_ark} scp:{src_scp} ark,scp:{dir}/normalized.ark,{dir}/normalized.scp'.format(
            args=args,
            cmvn_ark=cmvn_ark,
            src_scp=feat_scp,
            dir=os.path.dirname(feat_scp)
        )
        run_processes(cmd, environment=get_kaldi_env())


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
        with open(ark, 'rb') as ark_read_buffer:
            data = read_ark_buffer(ark_read_buffer)
        return data
    else:
        return read_ark_mat(ark, pos)


def read_ark_buffer(ark_buffer):
    data = dict()
    while True:
        utt_id = ''
        next_char = ark_buffer.read(1)
        if next_char == ''.encode():
            break
        else:
            c = struct.unpack('<s', next_char)[0]
        while c != ' '.encode():
            utt_id += c.decode('utf8')
            c = struct.unpack('<s', ark_buffer.read(1))[0]
        pos = ark_buffer.tell() - 1
        header = struct.unpack('<xccccbibi', ark_buffer.read(15))
        ark_buffer.seek(pos)
        data[utt_id] = _read_mat_from_buffer(ark_buffer)
        rows, colums = header[-3], header[-1]
        ark_buffer.seek(
            pos + 16 + (rows * colums * 4))
    return data


def _read_mat_from_buffer(buffer):
    header = struct.unpack('<cxcccc', buffer.read(6))
    if header[1] != "B".encode():
        raise ValueError("Input .ark file is not binary")

    if header[2] == b'C':
        raise ValueError("Input .ark file is compressed. You have to "
                         "decompress it first using copy-feats.")

    m, rows = struct.unpack('<bi', buffer.read(5))
    n, cols = struct.unpack('<bi', buffer.read(5))

    tmp_mat = np.frombuffer(
        buffer.read(rows * cols * 4), dtype=np.float32)
    utt_mat = np.reshape(tmp_mat, (rows, cols))
    return utt_mat


def read_ark_mat(ark, pos):
    with open(ark, 'rb') as ark_read_buffer:
        ark_read_buffer.seek(pos, 0)
        utt_mat = _read_mat_from_buffer(ark_read_buffer)
        ark_read_buffer.close()
    return utt_mat


def _import_alignment(ark, model_file, extract_cmd, extract_cmd_finish,
                      is_zipped=True):
    """ Read alignment data file.

        Can read either phones or pdfs depending on the copy_cmd.

        :param ark: The ark file to read
        :param model_file: Model file used to create the alignments. This is needed
            to extract the pdf ids
        :param extract_cmd: Command to extract the alignment. Can be either
            ali-to-pdf or ali-to-phones
        :param extract_cmd_finish: Success output of the extraction command
            (i.e. Done or Converted)
        :param copy_feats: The location of the kaldi tool `copy-feats`
        :return: A dictionary with the file ids as keys and their data as values
        """
    data = dict()
    if is_zipped:
        src_param = 'ark:gunzip -c {ark} |'.format(ark=ark)
    else:
        src_param = 'ark:{ark}'.format(ark=ark)
    dest_param = 'ark,t:-'
    copy_process = subprocess.Popen(
        [extract_cmd, model_file, src_param, dest_param],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=get_kaldi_env())
    out, err = copy_process.communicate()
    try:
        if copy_process.returncode != 0:
            raise ValueError("Returncode of{} was != 0. Stderr "
                             "output is:\n{}".format(extract_cmd, err))
        out = out.decode('utf-8')
        err = err.decode('utf-8')
        pos = err.find(extract_cmd_finish) + 1 + len(extract_cmd_finish)
        matrix_number = int(err[pos:].split()[0])
        for line in out.split('\n'):
            split = line.split()
            if len(split) > 0:
                utt_id = split[0]
                ali = np.asarray(split[1:], dtype=np.int32)
                data[utt_id] = ali
    except Exception as e:
        print('Exception during reading the alignments: {}'.format(e))
        print('Stderr: {}'.format(err))
    assert len(data) == matrix_number, \
        '{cmd} converted {num_matrix} alignments, ' \
        'but we read {num_data}'. \
            format(cmd=extract_cmd,
                   num_matrix=matrix_number, num_data=len(data))
    return data


def import_alignment_data(ark, model_file, is_zipped=True):
    """Import alignments as pdf ids

    Since the binary form is not documented and may change in future release,
    a kaldi tool (ali-to-pdf) is used to first create a ark file in text mode.

    :param ark: The ark file to read
    :param model_file: Model file used to create the alignments. This is needed
        to extract the pdf ids
    :return: A dictionary with the file ids as keys and their data as values
    """
    _cmd = 'ali-to-pdf'
    return _import_alignment(ark, model_file, _cmd, 'Converted', is_zipped)


def import_phone_alignment_data(ark, model_file, is_zipped=True):
    """Import alignments as phone ids

    Since the binary form is not documented and may change in future release,
    a kaldi tool (ali-to-pdf) is used to first create a ark file in text mode.

    :param ark: The ark file to read
    :param model_file: Model file used to create the alignments. This is needed
        to extract the pdf ids
    :return: A dictionary with the file ids as keys and their data as values
    """
    _cmd = 'ali-to-phones'
    return _import_alignment(ark, model_file, _cmd, 'Done', is_zipped)


def _write_array_for_kaldi(utt_id, array, fid, close_stream=False):
    utt_mat = np.ascontiguousarray(array, dtype=np.float32)
    rows, cols = utt_mat.shape
    fid.write(
        struct.pack('<%ds' % (len(utt_id)), utt_id.encode()))
    ark_pos = fid.tell()
    fid.write(
        struct.pack('<cxcccc',
                    ' '.encode(),
                    'B'.encode(),
                    'F'.encode(),
                    'M'.encode(),
                    ' '.encode()))
    fid.write(struct.pack('<bi', 4, rows))
    fid.write(struct.pack('<bi', 4, cols))
    fid.write(utt_mat)
    if close_stream:
        fid.close()
    return ark_pos + 1


class ArkWriter():
    def __init__(self, ark_filename):
        self.ark_path = ark_filename

    def __enter__(self):
        self.ark_fid = open(self.ark_path, "wb")
        self.scp = dict()
        return self

    def write_array(self, utt_id, array):
        ark_pos = _write_array_for_kaldi(utt_id, array, self.ark_fid)
        self.scp[utt_id] = self.ark_path + ':{}'.format(ark_pos)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ark_fid.close()
        if self.ark_path.endswith('.ark'):
            write_scp_file(self.scp, self.ark_path.replace('.ark', '.scp'))
        else:
            write_scp_file(self.scp, self.ark_path + '.scp')


def array_to_kaldi_io_stream(array, utt_id='stream'):
    stream = BytesIO()
    _write_array_for_kaldi(utt_id, array, stream)
    return stream


def import_feat_scp(feat_scp, verbose=False):
    feat_scp = read_scp_file(feat_scp)
    utt_ids = list(feat_scp.keys())
    data_dict = dict()
    if verbose:
        print('Reading {} features'.format(len(utt_ids)))
        for utt_id in tqdm.tqdm(utt_ids):
            data_dict[utt_id] = import_feature_data(feat_scp[utt_id])
    else:
        for utt_id in utt_ids:
            data_dict[utt_id] = import_feature_data(feat_scp[utt_id])
    return data_dict


def read_scp_file(scp_file):
    """ Reads a scp file into a dict
    :param scp_file:
    :return:
    """
    if isinstance(scp_file, dict):
        return scp_file
    scp_feats = dict()
    with open(scp_file) as fid:
        lines = fid.readlines()
    for line in lines:
        scp_feats[line.split()[0]] = ' '.join(line.split()[1:])
    return scp_feats


def write_scp_file(scp, scp_file):
    with open(scp_file, 'w') as fid:
        for utt, val in scp.items():
            fid.write('{} {}\n'.format(utt, val))


def merge_scps(scp_1, scp_2, postfix_1=None, postfix_2=None):
    merged_scp = dict()
    if postfix_1 is None and postfix_2 is None:
        assert len(set(list(scp_1.keys()) + list(scp_2.keys()))) == \
               len(scp_1.keys()) + len(scp_2.keys())
    if postfix_1 is None:
        postfix_1 = ''
    if postfix_2 is None:
        postfix_2 = ''
    merged_scp.update({k + postfix_1: v for k, v in scp_1.items()})
    merged_scp.update({k + postfix_2: v for k, v in scp_2.items()})
    return merged_scp


window_type = "hamming"


def read_trans_file(trans_file):
    transcriptions = dict()
    with open(trans_file) as fid:
        for line in fid:
            utt_id, trans = line.split('\t')
            transcriptions[utt_id] = trans.strip()
    return transcriptions


def read_text_file(text_file):
    transcriptions = dict()
    with open(text_file, encoding='utf-8') as fid:
        for line in fid:
            utt_id = line.split()[0]
            transcriptions[utt_id] = ' '.join(line.split()[1:]).strip()
    return transcriptions


def import_alignment(ali_dir, model_file=None):
    """ Imports an alignments (pdf-ids)

    :param ali_dir: Directory containing the ali.* files
    :param model_file: Model used to create the alignments
    :return: Dict with utterances as key and alignments as value
    """

    if model_file is None:
        model_file = os.path.join(ali_dir, 'final.mdl')
    data_dict = dict()
    print('Importing alignments')
    for file in os.listdir(ali_dir):
        if file.startswith('ali'):
            ali_file = os.path.join(ali_dir, file)
            data_dict.update(import_alignment_data(ali_file, model_file))
    return data_dict


def import_phone_alignment(ali_dir, model_file=None):
    """ Imports an alignments (phone-ids)

    :param ali_dir: Directory containing the ali.* files
    :param model_file: Model used to create the alignments
    :return: Dict with utterances as key and alignments as value
    """

    if model_file is None:
        model_file = os.path.join(ali_dir, 'final.mdl')
    data_dict = dict()
    print('Importing alignments')
    for file in os.listdir(ali_dir):
        if file.startswith('ali'):
            ali_file = os.path.join(ali_dir, file)
            data_dict.update(import_phone_alignment_data(ali_file, model_file))
    return data_dict


def import_features_and_alignment(feat_scp, ali_dir, model_file,
                                  ali_mapper=lambda x: x, cut_alignments=False,
                                  cut_features=False, ali_type='pdf-ids',
                                  verbose=False):
    """ Import features and alignments given a scp file and a ali directory

    This is basically a wrapper around the other functions

    .. note:: This wrapper checks if the utterance ids match

    :param feat_scp: Scp file describing the features
    :param ali_dir: Directory with alignments
    :param model_file: Model used to create alignments
    :return: Tuple of (dict, dict) where the first contains the features and
        the second the corresponding alignments
    """
    features = import_feat_scp(feat_scp, verbose=verbose)
    if ali_type == 'pdf-ids':
        alignments = import_alignment(ali_dir, model_file)
    elif ali_type == 'phones':
        alignments = import_phone_alignment(ali_dir, model_file)
    else:
        raise ValueError('Unknown alignment type {}. Possible are '
                         'pdf-ids or phones.'.format(ali_type))

    features_common = dict()
    alignments_common = dict()
    for utt_id in features.keys():
        if ali_mapper(utt_id) in alignments.keys():
            features_common[utt_id] = features[utt_id]
            alignments_common[ali_mapper(utt_id)] = \
                alignments[ali_mapper(utt_id)]
            if ali_type != 'phones':
                len_features = features_common[utt_id].shape[0]
                len_ali = alignments_common[ali_mapper(utt_id)].shape[0]
                if cut_features:
                    features_common[utt_id] = features_common[utt_id][:len_ali]
                if cut_alignments:
                    alignments_common[ali_mapper(utt_id)] = \
                        alignments_common[ali_mapper(utt_id)][:len_features]
                if features_common[utt_id].shape[0] != \
                        alignments_common[ali_mapper(utt_id)].shape[0]:
                    warnings.warn(
                        'There are {} features for utterance {} but '
                        '{} alignments'.format(
                            features_common[utt_id].shape[0], utt_id,
                            alignments_common[ali_mapper(utt_id)].shape[0]
                        ))
                    features_common.pop(utt_id)
                    alignments_common.pop(utt_id)
        else:
            warnings.warn('No alignment found for utterance {}'.format(utt_id))

    return features_common, alignments_common


def audioread_scp(scp, utt_ids, offset=0, duration=None, sample_rate=16000):
    """ Converts audio from scp to wav files with Kaldi tools and reads them with audioread.

    :param scp: .scp file to read audio from
    :param utt_ids: either list of utterance ids or one utterance id, which should be loaded
    :param offset: Begin of loaded audio
    :param duration: Durations of loaded audio
    :param sample_rate: Sample rate of audio
    :return: A dict which maps utterance ids to corresponding audio (when utt_ids is given as a list)
    or an array containing the audio data of one single utterance (when utt_ids is a single id)
    """
    scp = read_scp_file(scp)
    cmds = list()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Convert to .wav files into temp_dir
        if not isinstance(utt_ids, str):
            for utt_id in utt_ids:
                cmds.append(scp[utt_id][:len(
                    scp[utt_id]) - 1] + "> " + tmp_dir + "/" + utt_id)
        else:
            cmds.append(scp[utt_ids][:len(
                scp[utt_ids]) - 1] + "> " + tmp_dir + "/" + utt_ids)

        run_processes(cmds, environment=get_kaldi_env())

        # read audio
        if not isinstance(utt_ids, str):
            audio = dict()
            for utt_id in utt_ids:
                audio[utt_id] = audioread(tmp_dir + "/" + utt_id, offset,
                                          duration, sample_rate)
        else:
            audio = audioread(tmp_dir + "/" + utt_ids, offset, duration,
                              sample_rate)

        return audio


def make_fbank_features_from_time_signal(time_signal, num_mel_bins,
                                         low_freq=20, high_freq=-400,
                                         add_deltas=True, use_energy=False,
                                         use_log_fbank=True,
                                         window_type="povey"):
    audio_data = BytesIO()
    audiowrite(time_signal, audio_data, normalize=True, threaded=False)

    cmd_template = _build_cmd(RAW_FBANK_ARK_CMD, add_deltas=add_deltas)

    cmd = cmd_template.format(
        num_mel_bins=num_mel_bins,
        low_freq=low_freq, high_freq=high_freq,
        use_energy=use_energy, use_log_fbank=use_log_fbank,
        window_type=window_type
    )

    p = subprocess.Popen(cmd,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=False, shell=True,
                         env=get_kaldi_env(), bufsize=0)

    p.stdin.write(b'utt ')
    std, stderr = p.communicate(input=audio_data.getvalue())
    kaldi_data = BytesIO(std)
    return read_ark_buffer(kaldi_data)['utt']
