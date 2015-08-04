import numpy

def create_dict_from_transcriptions(FileList, set, transcript_type):
    """ Create dictionary for label to int mapping of transcriptions

    :param FileList: Dictionary with label sequence for each type, set and recording id:
                     FileList[transcript_type][set][recording_id]['label']
    :param set: set to be used, e.g. 'test', 'train', 'eval'
    :param transcript_type: type of transcription, e.g. 'phoneme_transcription', 'word_transcription'
    :return: Dictionary for label to int mapping: Dictionary[label] = int_label
    """
    LabelToInt = dict()
    for RecordingId in FileList[transcript_type][set]:
        for Label in FileList[transcript_type][set][RecordingId]['label']:
            if Label not in LabelToInt:
                LabelToInt[Label] = len(LabelToInt)
    return LabelToInt

def get_files_and_transciptions(FileList, set, transcript_type, file_type='audio'):
    """

    :param FileList: Dictionary with label sequence and file names for each type, set and recording id:
                     FileList[transcript_type][set][recording_id]['label']
                     FileList[file_type][set][recording_id]
    :param set: set to be used, e.g. 'test', 'train', 'eval'
    :param transcript_type: type of transcription, e.g. 'phoneme_transcription', 'word_transcription'
    :param file_type: type of files to read, e.g. 'audio'
    :return: List of filenames, list of transcriptions, list of recording ids
    """
    Ids = list(FileList[file_type][set].keys())
    Files = [FileList[file_type][set][Id] for Id in Ids]
    Transcriptions = [FileList[transcript_type][set][Id] for Id in Ids]

    return Files, Transcriptions, Ids

class CharLabelHandler():
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