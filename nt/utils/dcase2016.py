from nt.nn.data_fetchers import ArrayDataFetcher
from nt.utils.transcription_handling import EventLabelHandler
from nt.nn import DataProvider
from nt.nn.data_fetchers import JsonCallbackFetcher
import numpy as np
from nt.transform.module_fbank import logfbank
from nt.transform.module_mfcc import mfcc
from nt.utils.numpy_utils import add_context
import librosa
from nt.speech_enhancement.noise import set_snr
from nt.transform import module_stft


cv_scripts = ['clearthroat113', 'cough151', 'doorslam022', 'drawer072', 'keyboard066', 'keysDrop031', 'knock070',
              'laughter134', 'pageturn075', 'phone079', 'speech055', 'clearthroat115', 'cough168', 'doorslam023',
              'drawer076', 'keyboard068',
              'keysDrop072', 'knock072', 'laughter144', 'pageturn081', 'phone082', 'speech056']

def _generate_scripts(json_data):
    # generates training scripts
    train_scripts = list()
    events_list = json_data['train']['Complete Set']['annotation']['events']
    for key in events_list.keys():
        if key not in cv_scripts:
            train_scripts.append(key)
    return train_scripts


def make_input_arrays(json_data, flist, **kwargs):
    """Returns unstacked and normalized features """

    fetcher = JsonCallbackFetcher(
        'fbank',
        json_data,
        flist,
        transform_features,
        feature_channels=['observed/ch1'],
        transform_kwargs=kwargs
    )
    scripts = list()
    cv_data = list()
    train_data = list()
    silence_lengths = dict()
    for idx in range(len(fetcher)):
        # fetch the transformed data
        data = fetcher.get_data_for_indices((idx,))
        scripts.append(fetcher.utterance_ids[idx])
        # Randomize the amount of silence appended
        num_silent_rows = np.random.random_integers(0,
                                                    300)  # 300= randomly choosing a higher number > largest number of data frames in any event file
        silence_lengths.update({fetcher.utterance_ids[idx]: num_silent_rows})
        appending_silence = np.zeros((num_silent_rows, data['observed'].shape[1]))
        data['observed'] = np.concatenate((data['observed'], appending_silence)).astype(np.float32)

        if fetcher.utterance_ids[idx] in cv_scripts:
            cv_data.append(data['observed'])
        else:
            train_data.append(data['observed'])

    # This data is composed of both signal and silences!
    train_data = np.concatenate(train_data, axis=0)
    cv_data = np.concatenate(cv_data, axis=0)

    # Normalize training and cv data (Per feature basis) -->> yields much BETTER results
    training_mean = np.mean(train_data, axis=0)
    training_var = np.var(train_data, axis=0)
    train_data = (train_data - training_mean) / np.sqrt(training_var)
    cv_data = (cv_data - training_mean) / np.sqrt(training_var)

    ## Append training data with all 5 SNRs
    noisy_train_data = list()
    for snr in [-10, -6, 0, 6, 10]:
        ## Generate white noise and set SNR for training data
        noise = np.random.normal(0, 1, train_data.shape)
        set_snr(train_data, noise, snr)
        noisy_train_data.append((train_data + noise).astype(np.float32))
    noisy_train_data = np.concatenate(noisy_train_data)

    ## Append cv data with random SNRs from amongst the list, 5 times
    noisy_cv_data = list()
    for i in range(5):  # range(1)
        snr = np.random.choice([-10, -6, 0, 6, 10])
        ## Generate white noise and set SNR for cv data
        noise = np.random.normal(0, 1, cv_data.shape)
        set_snr(cv_data, noise, snr)
        noisy_cv_data.append((cv_data + noise).astype(np.float32))
    noisy_cv_data = np.concatenate(noisy_cv_data)

    return noisy_train_data, noisy_cv_data, silence_lengths, scripts

def transform_features(data, **kwargs):
    num_fbanks = kwargs.get('num_fbanks', 26)
    num_mfcc_coeff = kwargs.get('num_mfcc_coeff', 13)
    logfbank_feat = logfbank(data['observed'][0], number_of_filters=num_fbanks).astype(np.float32)
    if num_mfcc_coeff > 0:
        mfcc_feat = mfcc(data['observed'][0], numcep=num_mfcc_coeff, number_of_filters=num_fbanks)
        delta_feat = librosa.feature.delta(mfcc_feat)
        delta_delta_feat = librosa.feature.delta(mfcc_feat, order=2)
        data['observed'] = np.concatenate((logfbank_feat, delta_feat, delta_delta_feat), axis=1)
    else:
        data['observed'] = logfbank_feat
    return data


def make_target_arrays(event_label_handler, transcription_list, resampling_factor, scripts, silence_lengths):
    cv_target_list = list()
    train_target_list = list()

    assert len(scripts) == len(silence_lengths.keys())
    for script in scripts:
        transcription_script = transcription_list[script]
        target = event_label_handler.label_seq_to_int_arr(transcription_script,
                                                          resampling_factor)  # 2d matrix of size #transcriptionFrames X 12
        target_noise = np.zeros((silence_lengths[script], target.shape[1]))
        target_noise[:, 0] = 1
        target = np.concatenate((target, target_noise))
        if script in cv_scripts:
            cv_target_list.append(target)
        else:
            train_target_list.append(target)

    # extend training and cv targets 5 times to match the 5 SNR corrupted input data

    train_data = np.concatenate(train_target_list).astype(np.float32)
    train_data = np.tile(train_data, (5, 1))  # np.tile(train_data, (1, 1))
    cv_data = np.concatenate(cv_target_list).astype(np.float32)
    cv_data = np.tile(cv_data, (5, 1))  #np.tile(cv_data, (1, 1))

    return train_data, cv_data


def get_train_cv_data_provider(json_data, flist, transcription_list, events,
                               resampling_factor=16 / 44.1, batch_size=32,
                               cnn_features=False, **kwargs):
    left_context = kwargs.get('left_context', 0)
    right_context = kwargs.get('right_context', 0)
    step_width = kwargs.get('step_width', 1)
    num_fbanks = kwargs.get('num_fbanks', 26)

    train_scripts = _generate_scripts(json_data)

    ### Load training and CV input data #######
    train_data, cv_data, silence_lengths, scripts = make_input_arrays(json_data, flist, **kwargs)

    T, F = train_data.shape
    train_data = add_context(
        train_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features
    )
    if cnn_features:
        T, B, C, H, W = train_data.shape
        train_data = train_data.reshape((T, C, H, W))
    else:
        train_data = train_data.reshape(T, -1)

    print(train_data.shape)

    ### Load training and CV targets #######
    event_label_handler = EventLabelHandler(transcription_list, events)
    train_target, cv_target = make_target_arrays(event_label_handler, transcription_list, resampling_factor,
                                                 scripts, silence_lengths)

    #print(train_target.shape, cv_target.shape)

    train_data_fetcher = ArrayDataFetcher('x', train_data)
    train_target_fetcher = ArrayDataFetcher('targets', train_target)

    dp_train = DataProvider((train_data_fetcher, train_target_fetcher), batch_size=batch_size, shuffle_data=True)

    T, F = cv_data.shape
    cv_data = add_context(
        cv_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features
    )
    if cnn_features:
        T, B, C, H, W = cv_data.shape
        cv_data = cv_data.reshape((T, C, H, W))
    else:
        cv_data = cv_data.reshape(T, -1)
    print(cv_data.shape)

    cv_data_fetcher = ArrayDataFetcher('x', cv_data)
    cv_target_fetcher = ArrayDataFetcher('targets', cv_target)

    dp_cv = DataProvider((cv_data_fetcher, cv_target_fetcher), batch_size=batch_size, shuffle_data=False)

    return dp_train, dp_cv


def _obtainTrainingParameters(json_data, flist_train, **kwargs):
    _, _, _, _, training_mean, training_var = make_input_arrays(json_data, flist_train, **kwargs)

    return training_mean, training_var

def make_input_test_arrays(json_data, flist, **kwargs):
    fetcher = JsonCallbackFetcher('fbank',
                                  json_data,
                                  flist,
                                  transform_features,
                                  feature_channels=['observed/ch1'],
                                  transform_kwargs=kwargs)
    scripts = list()
    dev_data = list()
    for idx in range(len(fetcher)):
        # fetch the transformed data
        data = fetcher.get_data_for_indices((idx,))
        scripts.append(fetcher.utterance_ids[idx])
        dev_data.append(data['observed'])
        # print(fetcher.utterance_ids[idx],data['observed'].shape)

    dev_data = np.concatenate(dev_data, axis=0).astype(np.float32)
    return dev_data, scripts


def make_target_test_arrays(event_label_handler, transcription_list, resampling_factor, scripts):
    target_list = list()
    for script in scripts:
        transcription_script = transcription_list[script]
        target_list.append(event_label_handler.label_seq_to_int_arr(transcription_script, resampling_factor))
    target_list = np.concatenate(target_list, axis=0).astype(np.bool)

    return target_list


def get_test_data_provider(json_data, flist_dev, transcription_list, events,
                           resampling_factor=16 / 44.1, batch_size=32, cnn_features=False, **kwargs):
    left_context = kwargs.get('left_context', 0)
    right_context = kwargs.get('right_context', 0)
    step_width = kwargs.get('step_width', 1)

    # Load Test input data
    dev_data, scripts = make_input_test_arrays(json_data, flist_dev, **kwargs)

    # Normalize the test data all at once with parameters of training data
    # flist_train = 'train/Complete Set/wav/mono'
    #training_mean, training_var = _obtainTrainingParameters(json_data, flist_train, **kwargs)

    # Normalize the test data
    training_mean = np.mean(dev_data, axis=0)
    training_var = np.var(dev_data, axis=0)

    dev_data = (dev_data - training_mean) / np.sqrt(training_var)

    T, F = dev_data.shape

    dev_data = add_context(
        dev_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features
    )

    if cnn_features:
        T, B, C, H, W = dev_data.shape
        dev_data = dev_data.reshape((T, C, H, W))
    else:
        dev_data = dev_data.reshape(T, -1)
    # print(dev_data.shape)

    # Load Test targets
    event_label_handler = EventLabelHandler(transcription_list, events)
    dev_target = make_target_test_arrays(event_label_handler, transcription_list, resampling_factor, scripts)

    dev_target_scripts = list()
    for script in scripts:
        dev_target_script = make_target_test_arrays(event_label_handler, transcription_list, resampling_factor,
                                                    [script])
        #print(script,dev_target_script.shape)
        dev_target_scripts.append(dev_target_script)

    dp_scripts = list()
    for i in range(len(dev_target_scripts)):
        if i == 0:
            data = dev_data[0:dev_target_scripts[i].shape[0]]
        else:
            a = list()
            for j in range(len(dev_target_scripts[0:i])):
                a.append(dev_target_scripts[j].shape[0])
            first_idx = np.sum(a)
            second_idx = first_idx + dev_target_scripts[i].shape[0]
            data = dev_data[first_idx:second_idx]

        dev_data_fetcher = ArrayDataFetcher('x', data)
        dev_target_fetcher = ArrayDataFetcher('targets', dev_target_scripts[i])
        dp_script = DataProvider((dev_data_fetcher, dev_target_fetcher), batch_size=batch_size, shuffle_data=False)
        dp_scripts.append(dp_script)

    return dp_scripts


def resample_and_convert_frame_to_seconds(frame_num, frame_size=512, frame_shift=160, resampling_factor=44.1 / 16,
                                          sampling_rate=44100):
    sample_num = module_stft._stft_frames_to_samples(int(resampling_factor * frame_num), frame_size, frame_shift)
    return '%.4f' % (sample_num / sampling_rate)


def generate_onset_offset_label(decoded_allFrames, event_id, event_label_handler, filename):
    on_off_label = list()
    i = 0
    while i < decoded_allFrames.shape[0] - 1:
        onset = resample_and_convert_frame_to_seconds(
            i + 1)  # To save frame no. which is 1 greater than the array index.
        label_now = decoded_allFrames[i]
        j = i + 1
        label_next = decoded_allFrames[j]
        while (label_next == label_now and j < decoded_allFrames.shape[0] - 1):
            j += 1
            label_next = decoded_allFrames[j]
        offset = resample_and_convert_frame_to_seconds(
            j)  # To save frame no. [not exceeded by 1 here as it is already greater than the offset index by 1]
        if label_now == 1:
            class_label = event_label_handler.int_to_label[event_id]  # To save
            on_off_label.append((onset, offset, class_label))  # save

        i = j
    return on_off_label
