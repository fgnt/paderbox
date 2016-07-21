""" A collection of (mostly undocumented) functions used in various projects.

Most of these functions are tailored for a specific need. However, you might
find something useful if you alter it a little bit.

"""

import os
import collections
import h5py
import numpy as np
from nt.database.chime import get_data_provider_for_flist

import nt.speech_enhancement.beamformer as bf
from nt.evaluation import input_sxr, output_sxr
from nt.evaluation.pesq import threaded_pesq as pesq
from nt.io.audiowrite import audiowrite
from nt.speech_enhancement.mask_estimation import estimate_IBM
from nt.speech_enhancement.noise import get_snr
from nt.transform.module_stft import istft
from nt.utils import Timer, mkdir_p
from nt.utils.math_ops import covariance


def get_beamform_results(
        Y,
        X,
        N,
        mask_X,
        mask_N,
        context_samples=0,
        stft_size=1024,
        stft_shift=256,
        calculate_PESQ=True,
        reference_channel=4
):
    """ A wrapper, if you want to calculate more than one beamforming result.

    :param Y: Mixed signal with dimensions T times M times F
    :param X: Source image with dimensions T times M times F
    :param N: Noise image with dimensions T times M times F
    :param mask_X: Source mask with dimensions T times F
    :param mask_N: Noise mask with dimensions T times F
    :param context_samples: Number of samples to be dropped after beamforming
    :param stft_size: ...
    :param stft_shift: ...
    :param calculate_PESQ: If true, external PESQ library will calculate score.
    :return:
    """

    # Translate from Neural Network to Beamformer conventions
    Y = Y.T
    if X is not None:
        X = X.T
        N = N.T
    mask_X = mask_X.T
    mask_N = mask_N.T

    assert Y.shape[0] == 513
    assert Y.dtype in (np.complex, np.complex64, np.complex128)

    phi_XX = covariance(Y, mask_X)
    phi_NN = covariance(Y, mask_N)

    assert phi_XX.shape == (513, 6, 6)

    W_gev = bf.get_gev_vector(phi_XX, phi_NN)
    W_pca = bf.get_pca_vector(phi_XX)
    W_mvdr = bf.get_mvdr_vector(W_pca, phi_NN)
    W_gev_ban = bf.blind_analytic_normalization(W_gev, phi_NN)

    algorithms = 'gev pca mvdr gev_ban'.split()
    results = dict()
    z = dict()

    def _istft(stft_signal):
        return istft(stft_signal.T, size=stft_size, shift=stft_shift)

    def _get_results(W):
        Y_bf = bf.apply_beamforming_vector(W, Y)
        y_bf = _istft(Y_bf)[context_samples:]
        if X is not None:
            X_bf = bf.apply_beamforming_vector(W, X)
            N_bf = bf.apply_beamforming_vector(W, N)
            x_bf = _istft(X_bf)[context_samples:]
            n_bf = _istft(N_bf)[context_samples:]
            SDR, SIR, SNR = output_sxr(
                x_bf[:, np.newaxis, np.newaxis],
                n_bf[:, np.newaxis, np.newaxis]
            )
        else:
            SDR, SIR, SNR = None, None, None
        return SDR, SIR, SNR, y_bf

    results['cond'] = np.linalg.cond(phi_NN)
    z['input'] = _istft(Y[:, reference_channel, :])[context_samples:]
    for alg in algorithms:
        SDR, _, SNR, y_bf = _get_results(locals()['W_{}'.format(alg)])
        results['SDR_' + alg] = SDR
        results['SNR_' + alg] = SNR
        z[alg] = y_bf

    # Input SXR (here only on channel 5)
    if X is not None:
        x_bf = _istft(X[:, reference_channel, :])[context_samples:]
        n_bf = _istft(N[:, reference_channel, :])[context_samples:]
        results['input_SDR'], _, results['input_SNR'] = input_sxr(
            x_bf[:, np.newaxis, np.newaxis],
            n_bf[:, np.newaxis, np.newaxis]
        )
        if calculate_PESQ:
            algorithms += ['input']
            pesq_result = pesq(len(z) * [x_bf], [z[alg] for alg in algorithms])
            for idx, alg in enumerate(algorithms):
                results['PESQ_' + alg] = pesq_result[idx][-1]

    return results, z


def get_masks_ito(utt_id, flist, ito_data):
    masks = ito_data[flist][utt_id]['m'][:].transpose((1, 0, 2))
    concentration = ito_data[flist][utt_id]['concentration'][:]
    counts = np.bincount(np.argmax(concentration, axis=0))
    target_mask = masks[:, np.argmax(counts), :]
    noise_mask = 1 - target_mask
    return target_mask, noise_mask


def get_masks_ito_oracle(utt_id, flist, ito_data, X, N):
    masks = ito_data[flist][utt_id]['m'][:].transpose((1, 0, 2))
    mask_X, mask_N = estimate_IBM(X[:, 4, :], N[:, 4, :])
    cos_distances = list()
    for mask_idx in range(masks.shape[1]):
        mask = masks[:, mask_idx, :]
        norm = np.linalg.norm(mask_X) + np.linalg.norm(mask)
        cos_distances.append(np.sum(mask.ravel() * mask_X.ravel()) / norm)
    target_mask = masks[:, np.argmax(cos_distances), :]
    noise_mask = 1 - target_mask
    return target_mask, noise_mask


def get_current_batch_info(dp):
    """ Return utterance index and utterance id for current batch

    :param dp:
    :type dp: nt.nn.DataProvider
    :return:
    """
    utt_idx = dp.current_observation_indices[0]
    utt_id = dp.fetchers[0].utterance_ids[utt_idx]
    return utt_idx, utt_id


def get_batch_for_utt_id(dp, utt_id):
    """ Returns the batch for a specific utterance id

    :param dp: Configured data provider
    :return:
    """

    utt_ids = dp.fetchers[0].utterance_ids
    idx = next((i for i, utt in enumerate(utt_ids) if utt == utt_id), None)
    if idx is None:
        raise ValueError('Could not find the provided utterance id in data '
                         'provider')
    data = dict()
    for fetcher in dp.fetchers:
        outputs = fetcher.get_data_for_indices((idx,))
        for name, output in zip(fetcher.outputs, outputs):
            data[name] = output
    return data


def get_available_utt_ids(dp):
    return dp.fetchers[0].utterance_ids


def update_result_for_batch(result_dict, y_dict, batch, utt_id, flist_name, nns,
                            ems):
    result_id = flist_name + '_' + utt_id
    if not result_id in result_dict:
        result_dict[result_id] = dict()
        y_dict[result_id] = dict()

    with Timer() as t:
        if 'simu' in flist_name and not 'et' in flist_name:
            X = batch['Test'][:, :6, :]
            N = batch['Test'][:, 6:, :]
            Y = X + N
            SNR_Y = get_snr(X, N)
            result_dict[result_id]['SNR_Y'] = SNR_Y
        else:
            Y = batch['Test']
            X = None
            N = None
            result_dict[result_id]['SNR_Y'] = None

        # Neural networks
        for name, nn in nns.items():
            nn.inputs.Y = nn.inputs.X = np.abs(Y)
            nn.decode()
            gamma_X = np.median(nn.outputs.mask_X_hat.num, axis=1)
            gamma_N = np.median(nn.outputs.mask_N_hat.num, axis=1)

            result, y = get_beamform_results(Y, X, N, gamma_X, gamma_N, name)
            result_dict[result_id].update(result)
            y_dict.update(y)

        # Oracle
        if 'simu' in flist_name and not 'et' in flist_name:
            result, y = get_beamform_results(Y, X, N, None, None,
                                             name='oracle', oracle=True)
            result_dict[result_id].update(result)
            y_dict.update(y)

        # Ito
        for name, data in ems.items():
            if X is not None:
                gamma_X, gamma_N = get_masks_ito_oracle(utt_id, flist_name,
                                                        data, X, N)
                result, y = get_beamform_results(Y, X, N, gamma_X, gamma_N,
                                                 name + '_oracle')
                result_dict[result_id].update(result)
                y_dict.update(y)
            gamma_X, gamma_N = get_masks_ito(utt_id, flist_name, data)
            result, y = get_beamform_results(Y, X, N, gamma_X, gamma_N, name)
            result_dict[result_id].update(result)
            y_dict.update(y)


def write_result_to_h5_file(batch, utt_id, flist_name, nns, ems, data_dir,
                            update=False, write_wav=False, wav_root=''):
    result_id = flist_name + '_' + utt_id
    file_name = '{}/{}.h5'.format(data_dir, result_id)
    mkdir_p(os.path.dirname(file_name))
    mode = 'w' if update else 'a'
    with h5py.File(file_name, mode) as f:
        with Timer() as t:
            if 'simu' in flist_name and not 'et' in flist_name:
                X = batch['X']
                N = batch['N']
                Y = X + N
                if not 'SNR_Y' in f:
                    SNR_Y = get_snr(X, N)
                    f.create_dataset('SNR_Y', data=SNR_Y)
            else:
                Y = batch['Y']
                X = None
                N = None
                SNR_Y = -999
                if not 'SNR_Y' in f:
                    f.create_dataset('SNR_Y', data=SNR_Y)

            def _write_audio(y, model_name):
                for beamformer in y:
                    name, cond = flist_name.split('/')[-1].split('_')
                    env = utt_id.split('_')[-1]
                    wav_dir = os.path.join(wav_root,
                                           'model_{}'.format(model_name),
                                           'beamformer_{}'.format(beamformer),
                                           '_'.join([name, env, cond]))
                    mkdir_p(wav_dir)
                    wav_file = os.path.join(wav_dir, utt_id.upper() + '.wav')
                    audiowrite(y[beamformer], wav_file, normalize=True)

            def _create_result(grp, model_name, gamma_X, gamma_N):
                grp.create_dataset('gamma_X', data=gamma_X)
                grp.create_dataset('gamma_N', data=gamma_N)
                result, y = get_beamform_results(Y, X, N, gamma_X, gamma_N, '')
                for name, data in result.items():
                    if data is not None:
                        grp.create_dataset(name.replace('__', '_'), data=data)
                for beamformer, data in y.items():
                    if data is not None:
                        grp.create_dataset(beamformer, data=data)
                if write_wav:
                    _write_audio(y, model_name)

            # Neural networks
            for model_name, nn in nns.items():
                if not model_name in f:
                    grp = f.create_group(model_name)
                else:
                    grp = f[model_name]
                if not model_name+'/gamma_X' in f:
                    nn.inputs.Y = nn.inputs.X = np.abs(Y)
                    nn.decode()
                    gamma_X = np.median(nn.outputs.mask_X_hat.num, axis=1)
                    gamma_N = np.median(nn.outputs.mask_N_hat.num, axis=1)
                    _create_result(grp, model_name, gamma_X, gamma_N)

            # Oracle
            if 'simu' in flist_name and not 'et' in flist_name:
                if not 'oracle' in f:
                    grp = f.create_group('oracle')
                    result, y = get_beamform_results(Y, X, N, None, None,
                                                     name='', oracle=True)
                    for name, data in result.items():
                        grp.create_dataset(name.replace('__', '_'), data=data)
                    for name, data in y.items():
                        grp.create_dataset(name, data=data)
                result, y = get_beamform_results(Y, X, N, None, None,
                                                     name='', oracle=True)
                if write_wav:
                    _write_audio(y, 'oracle')

            # Ito
            for model_name, model_data in ems.items():
                if not model_name in f:
                    if X is not None:
                        oracle_name = model_name + '_oracle'
                        grp = f.create_group(oracle_name)
                        gamma_X, gamma_N = get_masks_ito_oracle(
                            utt_id, flist_name, model_data, X, N)
                        grp.create_dataset('gamma_X', data=gamma_X)
                        grp.create_dataset('gamma_N', data=gamma_N)
                        result, y = get_beamform_results(
                            Y, X, N, gamma_X, gamma_N, '')
                        for name, data in result.items():
                            grp.create_dataset(
                                name.replace('__', '_'), data=data)
                        for name, data in y.items():
                            grp.create_dataset(name, data=data)

                    grp = f.create_group(model_name)
                    gamma_X, gamma_N = get_masks_ito(utt_id, flist_name,
                                                     model_data)
                    _create_result(grp, model_name, gamma_X, gamma_N)

        if not 'timer' in f:
            f.create_dataset('timer', data=t.msecs)


def get_result_for_flist_and_utt_id(flist, utt_id, nns, ems):
    dp = get_data_provider_for_flist(flist)
    batch = get_batch_for_utt_id(dp, utt_id)
    result_dict = dict()
    y_dict = dict()
    update_result_for_batch(result_dict, y_dict, batch, utt_id, flist, nns, ems)
    return result_dict, y_dict


def store_result_for_flist_and_utt_id(flist, utt_id, nns, ems, update=False):
    dp = get_data_provider_for_flist(flist)
    batch = get_batch_for_utt_id(dp, utt_id)
    write_result_to_h5_file(batch, utt_id, flist, nns, ems, 'results_data',
                            update)


def collect_results(flist, result_dir,
                    measure='PESQ', beamformer='gev'):
    assert measure in ['SNR', 'PESQ', 'cond'], 'Only PESQ/SNR/cond available'
    assert beamformer in ['pca', 'mvdr', 'gev', 'gev_ban'], 'Unknown beamformer'
    if measure == 'cond':
        res_str = measure + '_'
    else:
        res_str = '{}_{}'.format(measure, beamformer)
    dp = get_data_provider_for_flist(flist, lambda x: x)
    utt_ids = get_available_utt_ids(dp)
    results = dict()
    for utt_id in utt_ids:
        file_name = '{}_{}.h5'.format(flist, utt_id)
        try:
            with h5py.File(os.path.join(result_dir, file_name), 'r') as f:
                for name in f:
                    if name + '/' + res_str in f:
                        try:
                            results[name][utt_id] = f[name][res_str].value
                        except KeyError:
                            results[name] = dict()
                            results[name][utt_id] = f[name][res_str].value
        except OSError:
            print('Skipping file {}'.format(file_name))
    return results


def update_dict(d, u):
    """ Recursively update dict d with values from dict u.

    Args:
        d: Dict to be updated
        u: Dict with values to use for update

    Returns: Updated dict

    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            default = v.copy()
            default.clear()
            r = update_dict(d.get(k, default), v)
            d[k] = r
        else:
            d[k] = v
    return d
