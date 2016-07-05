# This file contains functions and helper functions for polyphonic augmentation of training data
# and writing the extracted features and targets in an hdf5 file.
# This hdf5file can be then used by an hdf5 fetcher.

import librosa
import os
import h5py
import numpy as np
from nt.io.audiowrite import audiowrite
from nt.utils.numpy_utils import segment_axis
from nt.transform.module_fbank import logfbank
from nt.speech_enhancement.noise import NoiseGeneratorWhite

def generate_training_dict(dir_name):
    """
    Generate a dictionary of training data
    :param dir_name: Name of directory containing small utterances
    :return: Dictionary of scripts and their corresponding signal, sampling rate and list of scripts
    """
    y_data = dict()
    scripts = list()
    for file in os.listdir(dir_name):
        filename = ''.join((dir_name, file))
        scripts.append(file[:-4])
        y, sr = librosa.load(filename, sr=16000)
        y_data.update({file[:-4]: y})
    return y_data, sr, scripts


def generate_polyphonic_sound(dir_name, events, num_iterations=100, file_num=0, gp_name='train',
                              noisy=False):
    """
    This function generates wav files and their corresponding annotations
    :param dir_name: name of the directory containing DCASE training data. They are used as templates for polyphonic
     sound generation
    :param events: the list of events
    :param num_iterations: number of iterations to be run for a single file. Parameter that determines the length of
    the audio file
    :param file_num: Parameter to determine the name of the file being generated
    :param gp_name: file belongs to 'train' OR 'cv', again used for naming of the file
    :param noisy: determines whether to add noise to the signal or not.
    :return: time_signal and targets
    """
    y_data, sr, scripts = generate_training_dict(dir_name)
    sampling_rate = sr
    total_time_signal = list()
    total_target_list = list()
    seq_list = list()
    for i in range(num_iterations):
        time_signal = list()
        # assuming uniform distribution for maxpoly € (1,5)
        num_to_concat = np.random.random_integers(1, 5)
        event_numbers = np.random.choice(len(events), size=num_to_concat, replace=False)
        print(event_numbers)
        for event_number in event_numbers:
            event = events[event_number]
            # Pick any script from scripts that is the event under consideration
            nums = [n for n in range(len(scripts)) if scripts[n][:-3] == event]
            random_event = np.random.choice(nums, 1)[0]
            time_signal.append(y_data[scripts[random_event]])
        # Find offset from the very beginning
        if i == 0:
            offset = 0
        else:
            offset += total_time_signal[-1].size

        a = list()  # list of number of samples preceding an overlapping script
        for j in range(len(time_signal)):
            a.append(np.where(j == 0, 0, np.random.random_integers(0, time_signal[j - 1].size)))
            # "Absolute" start_time, end_time, event name
            seq_list.append((a[-1] + offset, a[-1] + time_signal[j].size + offset,
                             events[event_numbers[j]]))
            start = (a[-1] + offset) / sampling_rate
            end = (a[-1] + time_signal[j].size + offset) / sampling_rate
            class_label = events[event_numbers[j]]
        max_size = np.max([time_signal[k].size + a[k] for k in range(len(time_signal))])

        targets = np.zeros((max_size, len(events) + 1))
        for j in range(len(event_numbers)):
            targets[a[j]:a[j] + time_signal[j].size, event_numbers[j] + 1] = 1

        time_signal = [np.pad(time_signal[l], (a[l], max_size - (a[l] + time_signal[l].size)), mode='constant')
                       for l in range(len(time_signal))]
        time_signal = sum(time_signal)
        num_silent_rows = np.random.random_integers(512, 5000)
        silence = np.zeros((num_silent_rows,))
        total_time_signal.append(np.concatenate((time_signal, silence)))
        target_silence = np.zeros((num_silent_rows, len(events) + 1))
        target_silence[:, 0] = 1
        total_target_list.append(np.concatenate((targets, target_silence)))
    total_time_signal = np.concatenate(total_time_signal)
    total_target_list = np.concatenate(total_target_list)
    # A check to see at least one event is active at any given time frame.
    assert all([row[:].all() for row in total_target_list]) == False

    if noisy:
        # Corrupt with randomly generarated noise
        n_gen = NoiseGeneratorWhite()
        noisy_time_signal = list()
        snr_values = np.random.choice([-12, -6, 0, 6, 12], size=1)
        for snr in snr_values:
            noisy_time_signal.append(total_time_signal + n_gen.get_noise_for_signal(total_time_signal, snr=snr))
        noisy_time_signal = np.concatenate(noisy_time_signal)
        return noisy_time_signal, total_target_list
    else:
        return total_time_signal, total_target_list


def write_to_hdf5(hdf_filename, dir_name, events, num_iterations, num_train_files, num_cv_files, noisy=False,
                  stft_size=512,
                  stft_shift=160):
    """
    This function writes to the hdf5 file, the fbank, delta and delta-delta features of the polyphonic sounds
    generated using the generate_polyphonic_sound function. It also stores the corresponding targets.

    :param hdf_filename: name of the hdf5 file that is to written to
    :param dir_name: name of the directory containing DCASE training data. They are used as templates for polyphonic
     sound generation
    :param events: the list of events
    :param num_iterations: number of iterations. This corresponds to the length of generated files
    :param num_train_files: number of training files to be generated
    :param num_cv_files: number of validation files to be generated
    :param noisy: Boolean. True if generated files are to be separately corrupted with noise
    :return: writes to hdf file. No return in particular
    """
    with h5py.File(hdf_filename, 'w') as hf:
        for gp_name in ['train', 'cv']:
            gp1 = hf.create_group(gp_name)
            num_files = np.where(gp_name == 'train', num_train_files, num_cv_files)
            for num_file in range(num_files):
                time_sig, targets = generate_polyphonic_sound(dir_name, events, num_iterations=num_iterations,
                                                              file_num=num_file,
                                                              gp_name=gp_name, noisy=noisy)
                # convert from samples to frames- activating an event if it's present at least in 50% of the frame
                frames_arr_train = list()
                for i in range(targets.shape[1]):  # i.e. for every event
                    time_signal_seg = segment_axis(targets[:, i], length=stft_size, overlap=stft_size - stft_shift,
                                                   end='pad')
                    frames_arr_train.append((np.sum(time_signal_seg, axis=1) > 0.5 * stft_size).reshape(-1, 1))
                frames_arr_train = np.concatenate(frames_arr_train, axis=1)
                # Extract features from time signal
                logfbank_feat = logfbank(time_sig, number_of_filters=40)
                delta_feat = librosa.feature.delta(logfbank_feat, axis=0, width=3)  # row-wise in our case
                delta_delta_feat = librosa.feature.delta(logfbank_feat, axis=0, order=2, width=3)
                T, F = logfbank_feat.shape
                assert logfbank_feat.shape[0] == frames_arr_train.shape[0]
                subgrp = gp1.create_group(str(num_file + 1))
                _ = subgrp.create_dataset('targets', data=frames_arr_train)
                subsubgrp = subgrp.create_group('features')
                _ = subsubgrp.create_dataset('fbanks', data=logfbank_feat)
                _ = subsubgrp.create_dataset('deltas', data=delta_feat)
                _ = subsubgrp.create_dataset('delta-deltas', data=delta_delta_feat)
