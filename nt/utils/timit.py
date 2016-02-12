from nt.nn.data_fetchers import JsonCallbackFetcher
from nt.nn import DataProvider
from nt.transform import stft


TIMIT_JSON_FILE = '/net/storage/database_jsons/timit.json'


def default_transform(data, **kwargs):
    return {
        'X': stft(data['observed']).transpose((1, 0, 2))
    }


def get_data_provider_for_flist(flist, callback_fcn=default_transform):
    feature_channels = ['observed/ch1']

    if flist == 'train':
        set_name = 'Complete Set'
    elif flist == 'test':
        set_name = 'Complete Test Set'  # or 'Core Test Set'
    else:
        raise ValueError('Unknown flist')

    fetcher = JsonCallbackFetcher(
        'timit_fetcher',
        json_src=TIMIT_JSON_FILE,
        flist='{}/{}/wav'.format(flist, set_name),
        callback_fcn=callback_fcn,
        feature_channels=feature_channels,
    )
    return DataProvider(
        (fetcher,), batch_size=1, shuffle_data=False, max_queue_size=30
    )
