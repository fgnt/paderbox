from nt.nn.data_fetchers import ArrayDataFetcher
from nt.utils.transcription_handling import EventLabelHandler
from nt.nn import DataProvider
from nt.nn.data_fetchers import JsonCallbackFetcher
import numpy as np
from nt.transform.module_fbank import logfbank
from nt.utils.numpy_utils import add_context
import librosa
from nt.transform import module_stft
import os
from nt.io.audiowrite import audiowrite
from nt.speech_enhancement.noise import NoiseGeneratorWhite
from math import ceil
from nt.utils.numpy_utils import segment_axis


def _samples_to_stft_frames(samples, size=512, shift=160):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.
    """
    # I changed this from np.ceil to math.ceil, to yield an integer result.
    return ceil((samples - size + shift) / shift)

cv_scripts = ['clearthroat113', 'cough151', 'doorslam022', 'drawer072', 'keyboard066', 'keysDrop031', 'knock070',
              'laughter134', 'pageturn075', 'phone079', 'speech055', 'clearthroat115', 'cough168', 'doorslam023',
              'drawer076', 'keyboard068',
              'keysDrop072', 'knock072', 'laughter144', 'pageturn081', 'phone082', 'speech056']


def generate_augmented_training_data(dir_name):
    train_time_signal = list()
    cv_time_signal = list()
    total_lengths = dict()
    scripts = list()

    for file in os.listdir(dir_name):
        scripts.append(file[:-4])
        filename = ''.join((dir_name, file))
        y, sr = librosa.load(filename, 16000)
        num_silent_rows = np.random.random_integers(512, 50000)
        silence = np.zeros((num_silent_rows,))
        total_samples = y.shape[0] + num_silent_rows
        total_lengths.update({file[:-4]: total_samples})

        if file[:-4] in cv_scripts:
            cv_time_signal.append(np.concatenate((y, silence)))
        else:
            train_time_signal.append(np.concatenate((y, silence)))

    cv_time_signal = np.concatenate(cv_time_signal)
    train_time_signal = np.concatenate(train_time_signal)

    n_gen = NoiseGeneratorWhite()
    noisy_train_signal = list()
    snr_values = np.random.choice([-12, -6, 0, 6, 12], size=5, replace=False)
    for snr in snr_values:
        noisy_train_signal.append(train_time_signal + n_gen.get_noise_for_signal(train_time_signal, snr=snr))
    noisy_train_signal = np.concatenate(noisy_train_signal)

    noisy_cv_signal = list()
    snr_values = np.random.choice([-12, -6, 0, 6, 12], size=5, replace=False)
    for snr in snr_values:
        noisy_cv_signal.append(cv_time_signal + n_gen.get_noise_for_signal(cv_time_signal, snr=snr))
    noisy_cv_signal = np.concatenate(noisy_cv_signal)

    return noisy_train_signal, noisy_cv_signal, total_lengths, scripts


def make_input_arrays(dir_name, **kwargs):
    """Extract features from the generated time signals """

    noisy_train_signal, noisy_cv_signal, total_lengths, scripts = generate_augmented_training_data(dir_name)
    transformed_train = transform_features(noisy_train_signal, **kwargs)
    transformed_cv = transform_features(noisy_cv_signal, **kwargs)
    return transformed_train, transformed_cv, total_lengths, scripts

def transform_features(data, **kwargs):
    num_fbanks = kwargs.get('num_fbanks', 26)
    delta = kwargs.get('delta', 0)
    delta_delta = kwargs.get('delta_delta', 0)

    logfbank_feat = logfbank(data, number_of_filters=num_fbanks)
    data = logfbank_feat
    if delta == 1:
        delta_feat = librosa.feature.delta(logfbank_feat, axis=0, width=3)  # row-wise in our case
        data = np.concatenate((data, delta_feat), axis=1)
    if delta_delta == 1:
        delta_delta_feat = librosa.feature.delta(logfbank_feat, axis=0, order=2, width=3)
        data = np.concatenate((data, delta_delta_feat), axis=1)
    return data.astype(np.float32)


def make_target_arrays(event_label_handler, transcription_list, resampling_factor, scripts, total_lengths, stft_size,
                       stft_shift):
    train_target_list = list()
    cv_target_list = list()

    assert len(scripts) == len(total_lengths.keys())
    for script in scripts:
        transcription_script = transcription_list[script]
        target = event_label_handler.label_seq_to_int_arr_samples(transcription_script,
                                                          resampling_factor)

        silence_samples = total_lengths[script] - target.shape[0]
        target_silence = np.zeros((silence_samples, target.shape[1]), dtype=np.int32)
        target_silence[:, 0] = 1
        if script in cv_scripts:
            cv_target_list.append(np.concatenate((target, target_silence)))
        else:
            train_target_list.append(np.concatenate((target, target_silence)))

    # extend training and cv targets 5 times to match the 5 SNR corrupted input data

    train_target = np.concatenate(train_target_list)
    train_target = np.tile(train_target, (5, 1))
    cv_target = np.concatenate(cv_target_list)
    cv_target = np.tile(cv_target, (5, 1))

    # convert from samples to frames- activating an event if it's present at least in 50% of the frame
    frames_arr_train = list()
    for i in range(train_target.shape[1]):  # i.e. for every event
        time_signal_seg = segment_axis(train_target[:, i], length=stft_size, overlap=stft_size - stft_shift, end='pad')
        frames_arr_train.append((np.sum(time_signal_seg, axis=1) > 0.5 * stft_size).reshape(-1, 1))
    frames_arr_train = np.concatenate(frames_arr_train, axis=1)

    frames_arr_cv = list()
    for i in range(cv_target.shape[1]):  # i.e. for every event
        time_signal_seg = segment_axis(cv_target[:, i], length=stft_size, overlap=stft_size - stft_shift, end='pad')
        frames_arr_cv.append((np.sum(time_signal_seg, axis=1) > 0.5 * stft_size).reshape(-1, 1))
    frames_arr_cv = np.concatenate(frames_arr_cv, axis=1)

    return frames_arr_train.astype(np.float32), frames_arr_cv.astype(np.float32)


def get_train_cv_data_provider(dir_name, stft_size, stft_shift, transcription_list, events,
                               resampling_factor=16 / 44.1, batch_size=32,
                               cnn_features=False, deltas_as_channel=False, **kwargs):
    left_context = kwargs.get('left_context', 0)
    right_context = kwargs.get('right_context', 0)
    step_width = kwargs.get('step_width', 1)

    # Load training and CV input data #######
    train_data, cv_data, total_lengths, scripts = make_input_arrays(dir_name, **kwargs)

    # Feature Normalize the training and cv data
    training_mean = np.mean(train_data, axis=0)
    training_var = np.var(train_data, axis=0)

    train_data = (train_data - training_mean) / np.sqrt(training_var)
    cv_data = (cv_data - training_mean) / np.sqrt(training_var)

    T, F = train_data.shape
    train_data = add_context(
        train_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features,
        deltas_as_channel=deltas_as_channel
    )
    if cnn_features:
        T, B, C, H, W = train_data.shape
        train_data = train_data.reshape((T, C, H, W))
    else:
        train_data = train_data.reshape(T, -1)

    # Load training and CV targets
    event_label_handler = EventLabelHandler(events)
    train_target, cv_target = make_target_arrays(event_label_handler, transcription_list, resampling_factor,
                                                 scripts, total_lengths, stft_size, stft_shift)

    train_data_fetcher = ArrayDataFetcher('x', train_data)
    train_target_fetcher = ArrayDataFetcher('targets', train_target)

    dp_train = DataProvider((train_data_fetcher, train_target_fetcher), batch_size=batch_size, shuffle_data=True)

    T, F = cv_data.shape
    cv_data = add_context(
        cv_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features,
        deltas_as_channel=deltas_as_channel
    )
    if cnn_features:
        T, B, C, H, W = cv_data.shape
        cv_data = cv_data.reshape((T, C, H, W))
    else:
        cv_data = cv_data.reshape(T, -1)

    cv_data_fetcher = ArrayDataFetcher('x', cv_data)
    cv_target_fetcher = ArrayDataFetcher('targets', cv_target)

    dp_cv = DataProvider((cv_data_fetcher, cv_target_fetcher), batch_size=batch_size, shuffle_data=False)

    return dp_train, dp_cv, training_mean, training_var

def transform_features_test(data, **kwargs):
    num_fbanks = kwargs.get('num_fbanks', 26)
    delta = kwargs.get('delta', 0)
    delta_delta = kwargs.get('delta_delta', 0)
    logfbank_feat = logfbank(data['observed'][0], number_of_filters=num_fbanks)
    data['observed'] = logfbank_feat
    if delta == 1:
        delta_feat = librosa.feature.delta(logfbank_feat, axis=0, width=3)  # row-wise in our case
        data['observed'] = np.concatenate((data['observed'], delta_feat), axis=1)
    if delta_delta == 1:
        delta_delta_feat = librosa.feature.delta(logfbank_feat, axis=0, order=2, width=3)
        data['observed'] = np.concatenate((data['observed'], delta_delta_feat), axis=1)
    return data


def make_input_test_arrays(json_data, flist, **kwargs):
    sample_rate = kwargs.get('sample_rate', 16000)
    fetcher = JsonCallbackFetcher('fbank',
                                  json_data,
                                  flist,
                                  transformation_callback=transform_features_test,
                                  sample_rate=sample_rate,
                                  feature_channels=['observed/ch1'],
                                  transformation_kwargs=kwargs)
    scripts = list()
    dev_data = list()
    for idx in range(len(fetcher)):
        # fetch the transformed data
        data = fetcher.get_data_for_indices((idx,))
        # Normalize every test sequence with it's own parameters
        seq_mean = np.mean(data['observed'], axis=0)
        seq_var = np.var(data['observed'], axis=0)
        data['observed'] = (data['observed'] - seq_mean) / np.sqrt(seq_var)
        scripts.append(fetcher.utterances[idx])
        dev_data.append(data['observed'])

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
                           resampling_factor=16 / 44.1, batch_size=32, cnn_features=False, deltas_as_channel=False,
                           **kwargs):
    left_context = kwargs.get('left_context', 0)
    right_context = kwargs.get('right_context', 0)
    step_width = kwargs.get('step_width', 1)

    # Load Test input data
    dev_data, scripts = make_input_test_arrays(json_data, flist_dev, **kwargs)

    T, F = dev_data.shape

    dev_data = add_context(
        dev_data.reshape(T, 1, F),
        left_context=left_context,
        right_context=right_context,
        step=step_width,
        cnn_features=cnn_features,
        deltas_as_channel=deltas_as_channel
    )

    if cnn_features:
        T, B, C, H, W = dev_data.shape
        dev_data = dev_data.reshape((T, C, H, W))
    else:
        dev_data = dev_data.reshape(T, -1)  # commenet this for LSTMs.
    print(dev_data.shape)

    # Load Test targets
    event_label_handler = EventLabelHandler(transcription_list, events)
    # dev_target = make_target_test_arrays(event_label_handler, transcription_list, resampling_factor, scripts)

    dev_target_scripts = list()
    for script in scripts:
        dev_target_script = make_target_test_arrays(event_label_handler, transcription_list, resampling_factor,
                                                    [script])
        print(dev_target_script.shape)
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

    return dp_scripts, scripts

def resample_and_convert_frame_to_seconds(frame_num, frame_size=512, frame_shift=160, resampling_factor=44.1 / 16,
                                          sampling_rate=44100):
    sample_num = module_stft._stft_frames_to_samples(frame_num, frame_size, frame_shift)
    # convert into samples corresponding to the frame size and shift
    return (resampling_factor * sample_num )/ sampling_rate # convert it into seconds

def generate_onset_offset_label(decoded_allFrames, event_id, event_label_handler, filename, garbage_events,
                                 frame_size=512, frame_shift=160,
                                resampling_factor=44.1 / 16):
    class_label = event_label_handler.int_to_label[event_id]
    i = 0
    file = open(filename, 'a')
    while i < decoded_allFrames.shape[0] - 1:
        onset = resample_and_convert_frame_to_seconds(
            i + 1, frame_size, frame_shift, resampling_factor)  # To save frame no. which is 1 greater than the array
        # index.
        label_now = decoded_allFrames[
            i]  # label_now and label_next are either 1 or 0 indicating the event to be active or inactive
        j = i + 1
        label_next = decoded_allFrames[j]
        while label_next == label_now and j < decoded_allFrames.shape[0] - 1:
            j += 1
            label_next = decoded_allFrames[j]
        offset = resample_and_convert_frame_to_seconds(
            j, frame_size, frame_shift,
            resampling_factor)  # To save frame no. [not exceeded by 1 here as it is already greater than the offset
        # index by 1]
        # if the period in question was active for that event, log it's onset and offset.
        # Minimum duration constraint
        if label_now == 1 and class_label != 'Silence' and class_label not in garbage_events and offset - onset > 0.06:
            file.write(''.join(('\t'.join(('%.2f' % onset, '%.2f' % offset, class_label)), '\n')))
        i = j
    file.close()
