import numpy as np
from itertools import combinations
import h5py, librosa, os
from nt.transform.module_fbank import logfbank
import math
from nt.utils.numpy_utils import segment_axis
from nt.speech_enhancement.noise import NoiseGeneratorWhite

background = ['washing machine', 'fridge', 'microwave running', 'ironing', 'coffee maker', 'vacuum cleaner',
              'music playing', 'television']

short_events = ['mouse', 'page turning', 'paper rustling', 'mouse rolling', 'mouse clicking', 'adjusting fabric',
                'adjusting thermostat', 'adjusting sheet', 'lid', 'dispenser drawer', 'switch', 'glass', 'cup',
                'pot', 'fridge door', 'cork', 'closing bottle', 'opening bottle', 'coffee pot', 'plate', 'garbage can',
                'carpet', 'pen clicking', 'paper shuffling', 'dishwasher rack', 'noise', 'unknown sound',
                'thumping', 'iron', 'scooping detergent', 'shaking container', 'typing', 'whooshing',
                'pan banging', 'scooping coffee', 'pressing button', 'microwave', 'rubbing', 'writing',
                'paper hitting', 'cable', 'bird singing', 'power cord']

human_reflexes = ['person breathing', 'person whistling', 'person clapping', 'person coughing', 'person sneezing',
                  'person swallowing', 'people speaking']

water_activities = ['water spraying', 'water pouring', 'pouring drink', 'sink draining', 'soda stream', 'water',
                    'water splashing', 'water dripping']

object_actions = ['(object) knocking', 'chair squeaking', '(object) wobbling', '(object) tapping',
                  '(object) sweeping', 'door closing', '(object) rattling', '(object) shaking', '(object) knocking',
                  '(object) dragging', '(object) squeaking', '(object) swiping', 'door opening', '(object) pulling',
                  '(object) clacking', '(object) ripping', '(object) rubbing', '(object) ticking', '(object) clicking',
                  '(object) beeping', '(object) jingling', '(object) scraping', '(object) opening',
                  '(object) shuffling']

# Generate a dictionary of training data
def generate_training_dict(dir_name):
    print('Generating sampled data from given utterances....')
    y_data = dict()
    scripts = list()
    for file in os.listdir(dir_name):
        filename = ''.join((dir_name, file))
        scripts.append(file[:-4])
        y, sr = librosa.load(filename, sr=16000)
        y_data.update({file[:-4]: y})
    return y_data, sr, scripts


def seconds_to_samples(timestamp_in_seconds, sampling_rate):
    return int(math.floor(timestamp_in_seconds * sampling_rate))


def get_targets_from_transcriptions(transcription_file, num_of_samples, events, sampling_rate, stretching_factor=1):
    num_events = len(events)
    fid = open(transcription_file, 'r')
    lines = fid.read().split('\n')
    fid.close()
    int_arr = np.zeros((num_of_samples, num_events + 1), dtype=np.int32)
    dict_events = generate_transcription(events)
    for line in lines[:-1]:
        start, end, class_label = line.split('\t')
        start = seconds_to_samples(float(start) * stretching_factor, sampling_rate)
        end = seconds_to_samples(float(end) * stretching_factor, sampling_rate)

        if class_label in background:
            class_label = 'background'
        elif class_label in short_events:
            class_label = 'short_events'
        elif class_label in human_reflexes:
            class_label = 'human_reflexes'
        elif class_label in water_activities:
            class_label = 'water_activities'
        elif class_label in object_actions:
            class_label = 'object_actions'
        int_arr[start:end, dict_events[class_label]] = 1

    return int_arr


def generate_transcription(events):
    label_to_int = dict()
    int_to_label = dict()
    label_to_int['Silence'] = 0
    int_to_label[0] = 'Silence'
    for i in range(len(events)):
        label_to_int[events[i]] = i + 1
        int_to_label[i + 1] = events[i]
    return label_to_int


def convert_to_frames(train_target, stft_size=512, stft_shift=160):
    print('Converting target samples to frames... ')
    # convert from samples to frames- activating an event if it's present at least in 50% of the frame
    frames_arr_train = list()
    for i in range(train_target.shape[1]):  # i.e. for every event
        time_signal_seg = segment_axis(train_target[:, i], length=stft_size, overlap=stft_size - stft_shift, end='pad')
        frames_arr_train.append((np.sum(time_signal_seg, axis=1) > 0.5 * stft_size).reshape(-1, 1))
    frames_arr_train = np.concatenate(frames_arr_train, axis=1)
    return frames_arr_train


def data_from_hdf5(hdf5_file):
    """ Reads features amd targets from a hdf file
    :param hdf5_file: hdf file to be read from
    :return: a tuple of features and targets 
    """

    with h5py.File(hdf5_file, 'r') as hf:
        targets = hf.get('/'.join(('evaluate', hdf5_file.split('/')[-1][5:9], 'targets')))
        targets = np.array(targets)
        fbanks = np.array(hf.get('/'.join(('evaluate', hdf5_file.split('/')[-1][5:9], 'features/fbanks'))))
        deltas = np.array(hf.get('/'.join(('evaluate', hdf5_file.split('/')[-1][5:9], 'features/deltas'))))
        delta_deltas = np.array(hf.get('/'.join(('evaluate', hdf5_file.split('/')[-1][5:9], 'features/delta-deltas'))))

    return (fbanks, deltas, delta_deltas), targets


def _generate_new_targets(targets_block1, targets_block2):
    """
    Performs an OR operation on two blocks of targets to produce a single block.

    :param targets_block1: block 1 containing targets
    :param targets_block2: block 2 containing targets
    :return: a combined mix of the two input blocks
    """
    new_targets_block = targets_block1 + targets_block2
    # Removing 'Silence' when none exists
    for i in range(new_targets_block.shape[0]):
        if new_targets_block[i, 1:].any():
            new_targets_block[i, 0] = 0

    return new_targets_block


def block_mixing(features, targets, b=20, p=2):
    """
    A data augmentation technique that divides the given features into finite number of small blocks with equal
    number of frames each. p blocks are added at a time to generate a new block, Read more about it from
    the paper: Recurrent neural networks for polyphonic sound event detection in real life recordings.
    OR master thesis of the first author
    :param features: array of normal features (without any augmentation) OR features needed for augmentation
    :param targets: corresponding array of normal targets
    :param b : total number of blocks
    :param: p : number of blocks mixed at a time
    :return: augmented features and targets
    """
    T = features.shape[0]  # total number of frames
    f_b = T // b  # number of frames per block. Discard the remaining frames
    # features for every block can be extracted using features_list[block_number-1*f_b: block_number*f_b]
    # (n) so the numer of different combinations for p blocks can be obtained from this binomial coeffecient
    # (p)
    # Use list(combinations(range(20),2)) -- which lists all the possible combinations
    list_of_combs = list(combinations(range(20), p))
    aug_features_list = list()
    aug_targets_list = list()
    for comb in list_of_combs:
        block_1, block_2 = comb
        features_block1 = features[block_1 * f_b: (block_1 + 1) * f_b]
        features_block2 = features[block_2 * f_b: (block_2 + 1) * f_b]
        aug_features_list.append(np.maximum(features_block1, features_block2))
        targets_block1 = targets[block_1 * f_b: (block_1 + 1) * f_b]
        targets_block2 = targets[block_2 * f_b: (block_2 + 1) * f_b]
        aug_targets_list.append(_generate_new_targets(targets_block1, targets_block2))

    aug_features_list = np.concatenate((aug_features_list), axis=0)
    aug_targets_list = np.concatenate((aug_targets_list), axis=0)
    return aug_features_list, aug_targets_list


def simple_addition(list_of_combs, time_signal, targets, thr_samples, noisy=False, noiseGen=NoiseGeneratorWhite()):
    targets_list = list()
    fbanks_list = list()

    for comb in list_of_combs:
        print('Adding the combination...', comb)
        features, new_targets = simple_addition_1combination(comb, time_signal, targets, thr_samples, noisy, noiseGen)
        targets_list.extend(new_targets)
        fbanks_list.extend(features)
    print('All combinations added.')
    return fbanks_list, targets_list


def simple_addition_1combination(comb, time_signal, targets, thr_samples, noisy=False, n_gen=NoiseGeneratorWhite()):
    new_signal = time_signal[comb[0]]
    new_targets = targets[comb[0]]
    for i in range(1, len(comb)):
        new_signal = new_signal + time_signal[comb[i]]
        new_targets = _generate_new_targets(new_targets, targets[comb[i]])

    if noisy:
        print('Adding noise')
        new_signal = add_noise(new_signal, n_gen)

    ## Appending logfbank features and corresponding targets for small utterances to the list
    logfbank_feat_list = list()
    utt_targets_list = list()
    if len(new_signal) > thr_samples:
        for ts in range(0, len(new_signal) // thr_samples):
            logfbank_feat = logfbank(new_signal[ts * thr_samples:(ts + 1) * thr_samples], number_of_filters=40)
            utt_targets = new_targets[ts * thr_samples:(ts + 1) * thr_samples, :]
            utt_targets = convert_to_frames(utt_targets)
            # Activating silence class where no class is activated.
            for i in range(utt_targets.shape[0]):
                if not utt_targets[i].any():
                    utt_targets[i][0] = 1
            logfbank_feat_list.append(logfbank_feat)
            utt_targets_list.append(utt_targets)

    return logfbank_feat_list, utt_targets_list


def add_noise(time_sig, noise_gen):
    noisy_time_signal = list()
    snr_values = np.random.choice([-6, 0, 6], size=1)
    for snr in snr_values:
        noisy_time_signal.append(time_sig + noise_gen.get_noise_for_signal(time_sig, snr=snr))
    noisy_time_signal = np.concatenate(noisy_time_signal)
    return noisy_time_signal
