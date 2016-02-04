import numpy as np

from nt.io.data_dir import database_jsons as database_jsons_dir
from nt.io.data_dir import testing as testing_dir
from nt.nn import DataProvider
from nt.nn.data_fetchers import JsonCallbackFetcher
from nt.transform.module_stft import stft


def main():
    common_setup = {
        'json_src': database_jsons_dir('chime.json'),
        'flist': 'train/A_database/flists/wav/channels_6/tr05_simu'
    }

    def callback_fcn(data_dict, **kwargs):
        stft_opt = {"size": 1024, "shift": 256, "window_length": None}
        return {key: stft(val, **stft_opt) for key, val in data_dict.items()}

    fetcher = JsonCallbackFetcher('fetcher'
                                  , feature_channels_glob=['N/CH*', 'X/CH*', 'observed/CH*']
                                  , callback_fcn=callback_fcn, **common_setup)

    provider = DataProvider(
        [fetcher],
        batch_size=1, max_queue_size=10, shuffle_data=False,
        sleep_time=1
    )

    ans = provider.get_data_for_indices_tuple((1,))
    Y, X, N = ans['observed'], ans['X'], ans['N']

    print('start')
    file = testing_dir('speech_enhancement', 'data', 'beamformer')
    np.savez(file, X=X, Y=Y, N=N)
    with np.load(file + '.npz') as data:
        X = data['X']
    provider.shutdown()
    print('end')


if __name__ == "__main__":
    main()
