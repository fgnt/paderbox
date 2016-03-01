from nt.nn.data_fetchers import ArrayDataFetcher
from nt.utils.transcription_handling import EventLabelHandler
from nt.nn import DataProvider
from nt.nn.data_fetchers import JsonCallbackFetcher
import numpy as np
from nt.transform.module_fbank import logfbank
from nt.utils.numpy_utils import stack_context


cv_scripts = ['clearthroat113', 'cough151', 'doorslam022', 'drawer072', 'keyboard066', 'keysDrop031', 'knock070',
              'laughter134', 'pageturn075', 'phone079', 'speech055', 'clearthroat115', 'cough168', 'doorslam023',
              'drawer076', 'keyboard068',
              'keysDrop072', 'knock072', 'laughter144', 'pageturn081', 'phone082', 'speech056']


def _generate_scripts(json_data):
    # generates training scripts
    train_scripts = list()
    # cv_scripts = list()
    events_list = json_data['train']['Complete Set']['annotation']['events']
    for key in events_list.keys():
        if key not in cv_scripts:
            train_scripts.append(key)
    return train_scripts


def make_input_arrays(json_data, flist, cv_scripts, **kwargs):

    fetcher = JsonCallbackFetcher(
        'fbank',
        json_data,
        flist,
        transform_features_callback_function,
        feature_channels=['observed/ch1'],
        transform_kwargs=kwargs
    )

    cv_data = list()
    train_data = list()
    for idx in range(len(fetcher)):
        data = fetcher.get_data_for_indices((idx,))
        if fetcher.utterance_ids[idx] in cv_scripts:
            cv_data.append(data['observed'])
        else:
            train_data.append(data['observed'])

    train_data = np.concatenate(train_data,axis =0)
    cv_data = np.concatenate(cv_data,axis=0)
    return train_data, cv_data


def transform_features_callback_function(data, **kwargs):
    left_context = kwargs.get('left_context', 0)
    right_context = kwargs.get('right_context', 0)
    step_width = kwargs.get('step_width', 1)

    data['observed'] = logfbank(data['observed'][0], number_of_filters=26).astype(np.float32)

    B, F = data['observed'].shape

    data['observed'] = stack_context(
        data['observed'].reshape(1, B, F),
        left_context=left_context,
        right_context=right_context,
        step_width=step_width
    ).reshape(B, -1)

    return data


def make_target_arrays(event_label_handler, transcription_list, resampling_factor, train_scripts, cv_scripts):
    cv_target_list = list()
    train_target_list = list()
    scripts = list()
    scripts.extend(train_scripts)
    scripts.extend(cv_scripts)
    for script in scripts:
        transcription_script = transcription_list[script]
        if script in cv_scripts:
            cv_target_list.append(event_label_handler.label_seq_to_int_arr(transcription_script,resampling_factor))
        else:
            train_target_list.append(event_label_handler.label_seq_to_int_arr(transcription_script,resampling_factor))

    train_data = np.concatenate(train_target_list).astype(np.bool)
    cv_data = np.concatenate(cv_target_list)
    return train_data, cv_data


def get_train_cv_data_provider(json_data, flist, transcription_list, events,
                               resampling_factor=16 / 44.1, batch_size=32, **kwargs):

    train_scripts = _generate_scripts(json_data)

    ### Load training and CV input data #######
    train_data, cv_data = make_input_arrays(json_data, flist, cv_scripts, **kwargs)

    ### Load training and CV targets #######
    event_label_handler = EventLabelHandler(transcription_list, events)
    train_target, cv_target = make_target_arrays(event_label_handler, transcription_list, resampling_factor,
                                               train_scripts, cv_scripts)

    # Normalize training data (Per feature basis) -->> yields much BETTER results
    training_mean = np.mean(train_data, axis=0)
    training_var = np.var(train_data, axis=0)
    for i in range(train_data.shape[0]):
        train_data[i] = (train_data[i] - training_mean) / training_var

    train_data_fetcher = ArrayDataFetcher('x', train_data, bins=[0, train_target.shape[0]], left_context=0,
                                          right_context=0, with_context=False)
    train_target_fetcher = ArrayDataFetcher('targets', train_target, with_context=False)

    dp_train = DataProvider((train_data_fetcher, train_target_fetcher), batch_size=batch_size, shuffle_data=True)

    # Normalize using the parameters of training data
    for i in range(cv_data.shape[0]):
        cv_data[i] = (cv_data[i] - training_mean) / training_var

    cv_data_fetcher = ArrayDataFetcher('x', cv_data, bins=[0, cv_target.shape[0]], left_context=0,
                                       right_context=0, with_context=False)
    cv_target_fetcher = ArrayDataFetcher('targets', cv_target, with_context=False)

    dp_cv = DataProvider((cv_data_fetcher, cv_target_fetcher), batch_size=batch_size, shuffle_data=True)

    return (dp_train, dp_cv)
