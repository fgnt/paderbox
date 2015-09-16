import os

from nt.nn.data_fetchers.chime_feature_data_fetcher import \
    ChimeFeatureDataFetcher
from nt.nn import DataProvider

CHIME_JSON = '/net/storage/2015/chime/chime_ref_data/data/json/chime.json'


def get_chime_data_provider_for_flist(flist, window_length=1024, stft_size=1024,
                                      stft_shift=256,
                                      use_context_for_real=True):
    if flist[:2] == 'tr':
        stage = 'train'
    elif flist[:2] == 'dt':
        stage = 'dev'
    elif flist[:2] == 'et':
        stage = 'test'
    else:
        raise ValueError('flist seems to have the wrong format. It should be '
                         'something like tr05_simu or et05_real.')
    flist = '{}/A_database/flists/wav/channels_6/{}'.format(stage, flist)
    if 'real' in flist and use_context_for_real:
        start_key = 'start'
        end_key = 'end'
        feature_channels = ['embedded/CH{}'.format(ch) for ch in range(1, 7)]
        annotations = flist.replace('flists/wav/channels_6', 'annotations')
        fetchers = ChimeFeatureDataFetcher('Y', CHIME_JSON, flist,
                                           feature_type='stft',
                                           feature_channels=feature_channels,
                                           annotations=annotations,
                                           audio_start_key=start_key,
                                           audio_end_key=end_key,
                                           context_length=5,
                                           window_length=window_length,
                                           stft_shift=stft_shift,
                                           stft_size=stft_size),
    if 'simu' in flist and not 'et' in flist:
        fetchers = ChimeFeatureDataFetcher('X', CHIME_JSON, flist,
                                           feature_type='stft',
                                           feature_channels=[
                                               'X/CH{}'.format(n) for n
                                               in range(1, 7)],
                                           window_length=window_length,
                                           stft_shift=stft_shift,
                                           stft_size=stft_size), \
                   ChimeFeatureDataFetcher('N', CHIME_JSON, flist,
                                           feature_type='stft',
                                           feature_channels=[
                                               'N/CH{}'.format(n) for n
                                               in range(1, 7)],
                                           window_length=window_length,
                                           stft_shift=stft_shift,
                                           stft_size=stft_size)
    else:
        feature_channels = ['observed/CH{}'.format(n) for n in range(1, 7)]
        fetchers = ChimeFeatureDataFetcher('Y', CHIME_JSON, flist,
                                           feature_type='stft',
                                           feature_channels=feature_channels,
                                           window_length=window_length,
                                           stft_shift=stft_shift,
                                           stft_size=stft_size),
    return DataProvider(fetchers, batch_size=1)


def parse_kaldi_chime_results(kaldi_exp):
    def obtain_one_or_many_result_filenames(res_dir):
        result_files = [result for result in os.listdir(res_dir) if
                        result.split('.')[-1] == 'result']
        return result_files

    def parse_experiment(result):
        parse_result = dict()
        for data_set in ['et', 'dt']:
            for condition in ['real', 'simu']:
                parse_key = '{data_set}05_{condition} WER'.format(
                    data_set=data_set, condition=condition)
                for line in result:
                    if 'best overall' in line[:len('best overall')]:
                        parse_result[
                            '{}_{}_LM'.format(data_set, condition)] = int(
                            line.split()[-1][:-1])
                    elif parse_key in line[:len(parse_key)]:
                        parse_result[
                            '{}_{}_avg'.format(data_set, condition)] = float(
                            line.split()[2][:-1])
                        parse_result[
                            '{}_{}_bus'.format(data_set, condition)] = float(
                            line.split()[4][:-1])
                        parse_result[
                            '{}_{}_caf'.format(data_set, condition)] = float(
                            line.split()[6][:-1])
                        parse_result[
                            '{}_{}_ped'.format(data_set, condition)] = float(
                            line.split()[8][:-1])
                        parse_result[
                            '{}_{}_str'.format(data_set, condition)] = float(
                            line.split()[10][:-1])
        return parse_result

    train_database = []
    for train_folder in os.listdir(kaldi_exp):
        if train_folder.startswith('tri3b') and \
                (train_folder.split('_')[2] =='multi'
                or train_folder.split('_')[2] == 'simu'):
            train_database.append({
                "train_folder": train_folder,
                "train_set": train_folder.split('_')[1],
                "train_name": '_'.join(train_folder.split('_')[3:]),
                "am": 'GMM'
            })
        elif train_folder == "tri3b_tr05_orig_clean":
            train_database.append({
                "train_folder": train_folder,
                "train_set": train_folder[6:10],
                "train_name": "orig_clean",
                "am": 'GMM'
            })
        elif train_folder.startswith('tri4a') and train_folder.split('_')[
            3] == 'multi' and train_folder.split('_')[-1] != 'i1lats':
            train_database.append({
                "train_folder": train_folder,
                "train_set": train_folder.split('_')[2],
                "train_name": '_'.join(train_folder.split('_')[4:]),
                "am": train_folder.split('_')[1].upper()
            })
        elif train_folder.startswith('tri4a') and train_folder.split('_')[
            3] == 'multi' and train_folder.split('_')[-1] == 'i1lats':
            train_database.append({
                "train_folder": train_folder,
                "train_set": train_folder.split('_')[2],
                "train_name": '_'.join(train_folder.split('_')[4:-2]),
                "am": train_folder.split('_')[1].upper() + '_SMBR'
            })

    decode_database = []
    for exp in train_database:
        result_filenames = obtain_one_or_many_result_filenames(
            os.path.join(kaldi_exp, exp["train_folder"]))
        for filename in result_filenames:
            exp2_entry = exp.copy()
            if filename.split('_')[2] == 'ken':
                exp2_entry["enhance_name"] = '_'.join(
                    filename.split('_')[4:]) + '_kenLM'
            else:
                exp2_entry["enhance_name"] = '_'.join(
                    filename.split('_')[2:])
            exp2_entry["enhance_name"] = exp2_entry["enhance_name"].replace(
                '.result', '')
            full_path = os.path.join(kaldi_exp, exp["train_folder"],
                                     filename)
            with open(full_path) as fid:
                result = fid.readlines()
            try:
                exp2_entry.update(parse_experiment(result))
            except Exception:
                print('Error parsing file {}'.format(filename))
            else:
                decode_database.append(exp2_entry)

    return decode_database
