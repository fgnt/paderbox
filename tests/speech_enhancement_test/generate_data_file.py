import numpy as np
from nt.nn import DataProvider
from nt.nn.data_fetchers import ChimeFeatureDataFetcher

def main():
    common_setup = {
        'json_src': '/net/ssd/2015/chime/data/json/chime.json',
        'flist': 'train/A_database/flists/wav/channels_6/tr05_simu',
        'stft_shift': 256
    }

    feature_channels = ['X/CH{}'.format(x) for x in range(1, 7)]
    fetcher_X = ChimeFeatureDataFetcher('X', feature_channels=feature_channels, **common_setup)
    feature_channels = ['N/CH{}'.format(x) for x in range(1, 7)]
    fetcher_N = ChimeFeatureDataFetcher('N', feature_channels=feature_channels, **common_setup)
    feature_channels = ['observed/CH{}'.format(x) for x in range(1, 7)]
    fetcher_Y = ChimeFeatureDataFetcher('Y', feature_channels=feature_channels, **common_setup)
    provider = DataProvider(
        [fetcher_X, fetcher_N, fetcher_Y],
        batch_size=1, max_queue_size=10, shuffle_data=False,
        sleep_time=1
    )

    provider.__iter__()
    ans = provider.__next__()
    Y, X, N = ans['Y'], ans['X'], ans['N']
    print('start')
    np.savez('data', X=X, Y=Y, N=N)
    with np.load('data.npz') as data:
        X = data['X']
    provider.shutdown()
    print('end')

if __name__ == "__main__":
    main()
