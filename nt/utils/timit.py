from nt.nn.data_fetchers import JsonCallbackFetcher
from nt.nn import DataProvider
from nt.transform import stft


def default_transform(data, **kwargs):
    return {
        'X': stft(data['observed']).transpose((1, 0, 2))
    }


def get_data_provider_for_flist(
        flist,
        callback_fcn=default_transform,
        json_file='/net/storage/database_jsons/timit.json',
        **kwargs):
    """

    To get a data provider, use this:
    >>> dp = nt.utils.timit.get_data_provider_for_flist('train')
    >>> dp.data_info

    To get a test batch:
    >>> batch = dp.test_run()

    If you want to iterate over all batches:
    >>> for batch in dp.iterate():
    >>>     print(batch.keys())
    >>>     break

    :param flist:
    :param callback_fcn:
    :param json_file:
    :param kwargs:
    :return:
    """
    feature_channels = ['observed/ch1']
    enable_cache = kwargs.pop('enable_cache', False)

    set_name = 'Complete Set'

    fetcher = JsonCallbackFetcher(
        'timit_fetcher',
        json_src=json_file,
        flist='{}/{}/wav'.format(flist, set_name),
        callback_fcn=callback_fcn,
        feature_channels=feature_channels,
        transform_kwargs=kwargs,
        enable_cache=enable_cache
    )
    return DataProvider(
        (fetcher,), batch_size=1, shuffle_data=False, max_queue_size=30
    )
