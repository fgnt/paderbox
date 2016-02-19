__author__ = 'walter'

import nt.utils.json_utils as ju
import os

def identity(x):
    """ identity function - returns a one element list containing the input parameter

    :param x: input parameter
    :return: a one element list [x] containing the input parameter
    """
    return [x]

class word_to_grapheme():
    """ Word to grapheme conversion - splits a word into its letters, leaving fixed words unchanged

    :param fixed_words: words which should not be split

    :return: a callable instance of this class which takes a word as input and outputs a one element
             list with the corresponding grapheme sequence
    """
    def __init__(self, fixed_words=[]):
        self.fixed_words = fixed_words

    def __call__(self, word):
        if word not in self.fixed_words:
            return [' '.join(list(word))]
        else:
            return [word]

class word_to_phoneme():
    """ Word to phoneme conversion - convert a word into one or multiple, phoneme sequences using a lexicon

    :param lexicon: dictionary with word to phoneme sequence mappings. The mappings can be a list of multiple sequences

    :return: a callable instance of this class which takes a word as input and outputs a list with
             one or multiple corresponding phoneme sequence
    """
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def __call__(self, word):
        phonemes = self.lexicon[word]
        if isinstance(phonemes, (list, tuple)):
            return phonemes
        else:
            return [phonemes]

def create_data_dir(database, flist, channels, tlist, data_dir, utt2spk_map=None, word_to_word=identity,
                    convert_command='{}', always_add_channel_name=False):
    """ Create a data directory with input files for kaldi

    :param database: dabase dictionary
    :param flist: path to audio file list
    :param channels: path to specific channels
    :param tlist: path to transcriptions list
    :param data_dir: data dir to write output to
    :param utt2spk_map: setup for utterance id to speaker id mapping
                        Example: utt2spk_map = {'pos': [0,1], 'split_key': '-', 'join_key': '-'}
    :param word_to_word: function to transform a single word
    :param convert_command: command used to convert input file to wav
                            Example: '/net/ssd/software/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav {}|sox - -r 16k -t wav - |'
    :param always_add_channel_name: always add channel name to recording id
    """

    files = ju.traverse_to_dict(database, flist)
    files_per_channel = {channel: ju.get_flist_for_channel(files, channel) for channel in channels}
    transcriptions = ju.traverse_to_dict(database, tlist)

    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'text'), 'w', encoding='utf-8') as fp_text, \
         open(os.path.join(data_dir, 'wav.scp'), 'w') as fp_wav_scp, \
         open(os.path.join(data_dir, 'utt2spk'), 'w') as fp_utt2spk:

        for recording_id in sorted(files.keys()):
            for channel in channels:
                if always_add_channel_name or len(channels) > 1:
                    recording_id_channel = '-'.join([recording_id, channel])
                else:
                    recording_id_channel = recording_id

                transcription = ' '.join(convert_transcription(transcriptions[recording_id], word_to_word)[0])
                fp_text.write('{} {}\n'.format(recording_id_channel, transcription))

                file = files_per_channel[channel][recording_id]
                scp_format = '{} ' + convert_command + '\n'
                fp_wav_scp.write(scp_format.format(recording_id_channel, file))

                speaker_id = utt2spk(recording_id_channel, **utt2spk_map)
                fp_utt2spk.write('{} {}\n'.format(recording_id_channel, speaker_id))

    utt2spk_to_spk2utt_cmd = 'cat {} | utt2spk_to_spk2utt.pl > {}'.format(os.path.join(data_dir, 'utt2spk'), os.path.join(data_dir, 'spk2utt'))
    os.system(utt2spk_to_spk2utt_cmd)


def utt2spk(recording_id, pos=None, split_key=None, join_key=''):
    """ Converts utterance id to speaker id

    :param recording_id: recording id (string)
    :param pos: array positions to join with 'join_key' after splitting with 'split_key'
                or list of tuples of start end indices to extract from recording_id and join with 'join_key'
    :param split_key key to split at
    :param join_key key to join with

    :return: speaker id
    """

    if split_key and pos:
        recording_id_split = recording_id.split(split_key)
        return join_key.join([recording_id_split[i] for i in pos])

    if pos:
        return join_key.join([recording_id[start:end] for start, end in pos])

    return recording_id

def convert_transcription(transcription, word_to_word=identity, word_to_units=identity):
    """ convert transcription to uniform output representation

    :param transcription: transcription to be processed
    :param word_to_word: function for word to word mapping
    :param word_to_units: function for word to unit mapping

    :return: word and unit transcription
    """

    if isinstance(transcription, str):
        transcription = transcription.split()

    words = [unit for word in transcription for unit in word_to_word(word)[0].split()]
    units = [word_to_units(word) for word in words]

    return words, units

def create_dict_dir_from_database(database, flists, tlist, dict_dir, silence_units, optional_silence, word_to_units=identity, word_to_word=identity):
    """ Create a dictionary directory with input files for kaldi

    :param database: dabase dictionary
    :param flists: list of paths to audio file lists
    :param tlist: path to transcriptions list
    :param dict_dir: dict dir to write output to
    :param silence_units: list of silence units
    :param optional_silence: optional silence unit
    :param word_to_units: function for word to unit mapping
    :param word_to_word: function for word to word mapping
    """

    # get file list
    files = {}
    for flist in flists:
        files.update(ju.traverse_to_dict(database, flist))
    transcriptions = ju.traverse_to_dict(database, tlist)

    # get word to unit dictionary
    word_to_unit_dict = {word: units_list for recording_id in files for word, units_list in zip(*convert_transcription(transcriptions[recording_id], word_to_word, word_to_units))}
    unit_set = set(unit for units_list in word_to_unit_dict.values() for units in units_list for unit in units.split())

    # create output directory
    os.makedirs(dict_dir, exist_ok=True)

    # write nonsilence, optional silence and silence units
    with open(os.path.join(dict_dir, 'nonsilence_phones.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(unit for unit in sorted(unit_set) if unit not in silence_units) + '\n')

    with open(os.path.join(dict_dir, 'silence_phones.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(sorted(silence_units)) + '\n')

    with open(os.path.join(dict_dir, 'optional_silence.txt'), 'w', encoding='utf-8') as fp:
        fp.write(optional_silence + '\n')

    # write lexicon
    with open(os.path.join(dict_dir, 'lexicon.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(word + '\t' + units for word, units_list in sorted(word_to_unit_dict.items()) for units in units_list) + '\n')

    # write extra questions
    with open(os.path.join(dict_dir, 'extra_questions.txt'), 'w', encoding='utf-8') as fp:
        fp.write(' '.join(sorted(silence_units)) + '\n')
        fp.write(' '.join(unit for unit in sorted(unit_set) if unit not in silence_units) + '\n')

def create_lattice_word_segmentation_from_text(textfile, database, tlist, segmentation_dir, word_to_word_ref=identity,
                                               word_to_units_ref=identity, word_to_word_res=identity,
                                               word_to_units_res=identity):
    """ Create input files for Lattice Word Segmentation from text file (reference or one best)

    :param textfile: texfile with resulting transcription, one per line with preceeding recording id
    :param database: dabase dictionary
    :param tlist: path to transcriptions list
    :param segmentation_dir: segmentation dir to write segmentation input to
    :param word_to_word_ref: function for word to word mapping (reference)
    :param word_to_units_ref: function for word to units mapping (reference)
    :param word_to_word_res: function for word to word mapping (result)
    :param word_to_units_res: function for word to units mapping (result)
    """

    transcriptions = ju.traverse_to_dict(database, tlist)

    # write text data
    os.makedirs(segmentation_dir, exist_ok=True)
    seen_ref_sequences = dict()
    with open(segmentation_dir + 'text.txt', 'w') as fp_text_file, \
         open(segmentation_dir + 'text.txt.ref', 'w') as fp_ref_file, \
         open(segmentation_dir + 'file_list.txt', 'w') as fp_file_list_file, \
         open(textfile) as fp_input_text_file:

        fp_file_list_file.write(segmentation_dir + 'text.txt\n')
        for line in fp_input_text_file:
            split_line = line.split()

            ref_transcription = convert_transcription(transcriptions[split_line[0]], word_to_word_ref, word_to_units_ref)

            # check for duplicate
            ref_sequence = ' '.join(word[0] for word in ref_transcription[1])
            if ref_sequence in seen_ref_sequences:
                seen_ref_sequences[ref_sequence].append(split_line[0])
                continue
            else:
                seen_ref_sequences[ref_sequence] = [split_line[0]]

            ref_transcription_joined = ' </unk> '.join(word[0] for word in ref_transcription[1])
            ref_transcription_joined += ' </unk> </s> </unk>'
            fp_ref_file.write(ref_transcription_joined + '\n')

            input_transcription = convert_transcription(split_line[1:], word_to_word_res, word_to_units_res)
            input_transcription_joined = ' '.join(word[0] for word in input_transcription[1])
            fp_text_file.write(input_transcription_joined + '\n')

def create_lattice_word_segmentation_from_lattices(latticefile, database, tlist, segmentation_dir,
                                                   word_to_word_ref=identity, word_to_units_ref=identity):
    """ Create input files for Lattice Word Segmentation from lattices

    :param latticefile: file with resulting lattices
    :param database: dabase dictionary
    :param tlist: path to transcriptions list
    :param segmentation_dir: segmentation dir to write segmentation input to
    :param word_to_word_ref: function for word to word mapping (reference)
    :param word_to_units_ref: function for word to units mapping (reference)
    """

    transcriptions = ju.traverse_to_dict(database, tlist)

    # write text data
    os.makedirs(segmentation_dir, exist_ok=True)
    seen_ref_sequences = dict()
    with open(segmentation_dir + 'text.txt.ref', 'w') as fp_ref_file, \
         open(segmentation_dir + 'file_list_htk.txt', 'w') as fp_file_list_file, \
         open(latticefile) as fp_input_text_file:

        for line in fp_input_text_file:
            recording_id = os.path.splitext(os.path.split(line)[1])[0]

            ref_transcription = convert_transcription(transcriptions[recording_id], word_to_word_ref, word_to_units_ref)

            # check for duplicate
            ref_sequence = ' '.join(word[0] for word in ref_transcription[1])
            if ref_sequence in seen_ref_sequences:
                seen_ref_sequences[ref_sequence].append(recording_id)
                continue
            else:
                seen_ref_sequences[ref_sequence] = [recording_id]

            ref_transcription_joined = ' </unk> '.join(word[0] for word in ref_transcription[1])
            ref_transcription_joined += ' </unk> </s> </unk>'
            fp_ref_file.write(ref_transcription_joined + '\n')

            fp_file_list_file.write(line)