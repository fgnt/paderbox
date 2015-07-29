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