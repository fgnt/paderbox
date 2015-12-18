import numpy

class CharLabelHandler(object):
    """ Handles transforming from chars to integers and vice versa

    """

    def __init__(self, transcription_list, add_blank=True):
        self.label_to_int = dict()
        self.int_to_label = dict()
        if add_blank:
            self.label_to_int['BLANK'] = 0
            self.int_to_label[0] = 'BLANK'
        for transcription in transcription_list:
            for char in transcription:
                if not char in self.label_to_int:
                    number = len(self.label_to_int)
                    self.label_to_int[char] = number
                    self.int_to_label[number] = char

    def label_seq_to_int_arr(self, label_seq):
        int_arr = numpy.empty(len(label_seq), dtype=numpy.int32)
        for idx, char in enumerate(label_seq):
            int_arr[idx] = self.label_to_int[char]
        return int_arr

    def int_arr_to_label_seq(self, int_arr):
        return ''.join([self.int_to_label[i] for i in int_arr])

    def print_mapping(self):
        for char, i in self.label_to_int.items():
            print('{} -> {}'.format(char, i))

    def __len__(self):
        return len(self.label_to_int)


class WordLabelHandler(object):
    """ Handles transforming from words to integers and vice versa

    """

    def __init__(self, transcription_list, add_blank=True, min_count=20):
        self.label_to_int = dict()
        self.int_to_label = dict()
        if add_blank:
            self.label_to_int['BLANK'] = 0
            self.int_to_label[0] = 'BLANK'
        self.label_to_int['<UNK>'] = len(self.label_to_int)
        self.int_to_label[len(self.int_to_label)] = '<UNK>'
        word_count = dict()
        for transcription in transcription_list:
            for word in transcription.split():
                try:
                    word_count[word] += 1
                except KeyError:
                    word_count[word] = 1
        for word, count in word_count.items():
            if count > min_count:
                number = len(self.label_to_int)
                self.label_to_int[word] = number
                self.int_to_label[number] = word

    def label_seq_to_int_arr(self, label_seq):
        int_arr = list()
        for word in label_seq.split():
            try:
                int_arr.append(self.label_to_int[word])
            except KeyError:
                int_arr.append(self.label_to_int['<UNK>'])
        return numpy.asarray(int_arr, dtype=numpy.int32)

    def int_arr_to_label_seq(self, int_arr):
        return ' '.join([self.int_to_label[i] for i in int_arr])

    def print_mapping(self):
        for char, i in self.label_to_int.items():
            print('{} -> {}'.format(char, i))

    def __len__(self):
        return len(self.label_to_int)


class HybridLabelHandler(object):
    """ Handles transforming from words/chars to integers and vice versa

        This Handler creates a unique integer for each word which appears at
        least ``min_count`` times in the corpus. Other words are transformed to
        their underlying character sequence and each character has its unique
        integer.

    """

    def __init__(self, transcription_list, add_blank=True, min_count=20):
        self.label_to_int = dict()
        self.int_to_label = dict()
        if add_blank:
            self.label_to_int['BLANK'] = 0
            self.int_to_label[0] = 'BLANK'
        self.label_to_int['<UNK>'] = len(self.label_to_int)
        self.int_to_label[len(self.int_to_label)] = '<UNK>'
        self.character_idxs = list()
        word_count = dict()
        for transcription in transcription_list:
            for word in transcription.split():
                try:
                    word_count[word] += 1
                except KeyError:
                    word_count[word] = 1
        for word, count in word_count.items():
            if count > min_count:
                if not word in self.label_to_int:
                    number = len(self.label_to_int)
                    self.label_to_int[word] = number
                    self.int_to_label[number] = word
            else:
                for c in word:
                    if c not in self.label_to_int:
                        number = len(self.label_to_int)
                        self.label_to_int[c] = number
                        self.int_to_label[number] = c
                        self.character_idxs.append(number)

    def label_seq_to_int_arr(self, label_seq):
        int_arr = list()
        for word in label_seq.split():
            try:
                int_arr.append(self.label_to_int[word])
            except KeyError:
                # We have an unknown word and now return its spelling
                for c in word:
                    try:
                        int_arr.append(self.label_to_int[c])
                    except KeyError:
                        int_arr.append(self.label_to_int['<UNK>'])
        return numpy.asarray(int_arr, dtype=numpy.int32)

    def int_arr_to_label_seq(self, int_arr):
        char_state = False
        ret_str = ''
        for idx in int_arr:
            if idx > len(self.label_to_int):
                raise ValueError('No label for index {}'.format(idx))
            if idx not in self.character_idxs:
                if char_state:
                    ret_str += '</w> ' + self.int_to_label[idx]
                    char_state = False
                else:
                    if len(ret_str) > 0:
                        ret_str += ' ' + self.int_to_label[idx]
                    else:
                        ret_str += self.int_to_label[idx]
            else:
                if char_state:
                    ret_str += self.int_to_label[idx]
                else:
                    if len(ret_str) > 0:
                        ret_str += ' <w>' + self.int_to_label[idx]
                    else:
                        ret_str += '<w>' + self.int_to_label[idx]
                    char_state = True
        return ret_str

    def print_mapping(self):
        for char, i in self.label_to_int.items():
            print('{} -> {}'.format(char, i))

    def __len__(self):
        return len(self.label_to_int)

def argmax_ctc_decode(int_arr, label_handler):
    """ Decodes a ctc sequence

    :param int_arr: sequence to decode
    :param label_handler: label handler
    :type label_handler: CharLabelHandler
    :return: decoded sequence
    """

    max_decode = numpy.argmax(int_arr, axis=1)
    decode = numpy.zeros_like(max_decode)
    idx_dec = 0
    for idx, n in enumerate(max_decode):
        if idx > 0 and not n == max_decode[idx - 1]:
            decode[idx_dec] = n
            idx_dec += 1
        elif idx == 0:
            decode[idx_dec] = n
            idx_dec += 1
    idx_seq = [c for c in decode if c != 0]
    sequence = label_handler.int_arr_to_label_seq(idx_seq)
    return sequence


class EventLabelHandler(object):
    """ Handles transforming from chars to integers and vice versa

    """

    def __init__(self, transcription_list, window_length=400,
                 stft_size=512, stft_shift=160):
        self.label_to_int = dict()
        self.int_to_label = dict()

        self.sample_to_frame_idx = lambda sample_idx: \
            sample_to_frame_idx(sample_idx, window_length, stft_shift)

        # set up mapping dictionaries
        # convention: the last element of the transcription list is the
        # length of the wave file, therefore slice it off
        for transcription_sequence in transcription_list.values():
            for _, _, label in transcription_sequence:
                if label not in self.label_to_int:
                    number = len(self.label_to_int)
                    self.label_to_int[label] = number
                    self.int_to_label[number] = label

    def label_seq_to_int_arr(self, transcription):
        # for event detection it is assumed that the transcription is list of
        # tuples of the scheme (begin, end, 'label'). The last element contains
        # the overall file length, therefore transcription[-1][1] is equal to
        # the sequence length in samples
        assert transcription[-1][2] == 'END'
        transcription_length_in_samples = transcription[-1][1]
        transcription_length_in_frames = self.sample_to_frame_idx(
            transcription_length_in_samples)
        number_of_events = len(self.label_to_int)

        int_arr = numpy.zeros(
            (transcription_length_in_frames, number_of_events),
            dtype=numpy.int32)
        for begin, end, label in transcription[:-1]:
            begin_frame, end_frame = [self.sample_to_frame_idx(n)
                                      for n in (begin, end)]
            int_arr[begin_frame:end_frame, self.label_to_int[label]] = 1
        return int_arr

    def int_arr_to_label_seq(self, int_arr):
        raise NotImplementedError('This feature is currently missing!')

    def print_mapping(self):
        for char, i in self.label_to_int.items():
            print('{} -> {}'.format(char, i))

    def __len__(self):
        return len(self.label_to_int)


def sample_to_frame_idx(sample_idx, frame_size, frame_shift):
    """ Calculate corresponding frame index for sample index
        :param sample_idx: sample index
    """
    start_offset = (frame_size - frame_shift)/2
    frame_idx = (sample_idx - start_offset)//frame_shift
    return max(0, frame_idx)
