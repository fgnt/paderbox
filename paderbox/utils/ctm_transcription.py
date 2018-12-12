import operator
import numpy as np
from paderbox.utils.dtw import dtw


class Evaluate:
    """
    Evaluation class for ctm
    """
    def __init__(self, res, ref, max_type='count'):
        """ initialize evaluator with result and reference ctm

        :param res: result ctm
        :param ref: reference ctm
        :param max_type: criterion do decide for assignment:
                         'duration' (max accumulative overlap)
        """
        self._res = res
        self._ref = ref
        self._res_ref = None
        self._ins_del_valid = None
        self._ins_del_sub_cor = None
        self._unique_ins_del_sub_cor = None
        self._ins_del_sub_cor_p_r_f_sum = None
        self._unique_ins_del_sub_cor_p_r_f_sum = None
        self._conf_mat = None
        self._reverse_conf_mat = None
        self._label_assignment = None
        self._unique_label_assignment = None
        self._max_type = max_type

    @property
    def res_ref(self):
        """ return ctm with max overlapping reference added to each entry

        :return: ctm with max overlapping reference added to each entry
        """
        if self._res_ref is None:
            self._res_ref = {file: allign_res_ref(res, self._ref[file])
                             for file, res in self._res.items()
                             if file in self._ref and len(self._ref[file]) > 0}

        return self._res_ref

    @property
    def ins_del_valid(self):
        """ Get insertions, deletions and valid pairs

        :return: list with indicators for each pair consisting of 'V' (valid),
                 'I' insertion and 'D' (deletion)
        """
        if self._ins_del_valid is None:
            self._ins_del_valid = {file: get_ins_del_valid(res_ref)
                                   for file, res_ref in self.res_ref.items()}

        return self._ins_del_valid

    @property
    def conf_mat(self):
        """ Get confusion matrix

        :return: confusion matrix
        """
        if self._conf_mat is None:
            self._conf_mat = get_conf_mat(self.res_ref, self.ins_del_valid)

        return self._conf_mat

    @property
    def label_assignment(self):
        """ get label to ref assignment

        :return: label assignment list with pairs
        """
        if self._label_assignment is None:
            self._label_assignment = get_label_assignment(self.conf_mat,
                                                          self._max_type)

        return self._label_assignment

    @property
    def unique_label_assignment(self):
        """ get best unique label to ref assignment

        :return: label assignment list with paris
        """
        if self._unique_label_assignment is None:
            self._unique_label_assignment, self._reverse_conf_mat =\
                get_unique_label_assignment(self.label_assignment,
                                            self._max_type)

        return self._unique_label_assignment

    @property
    def ins_del_sub_cor(self):
        """ return insertions, deletions, substitutions and correct pairs

        :return: list with indicators for each pair consisting of 'C' (correct),
                 'I' (insertion), 'D' (deletion) and 'S' (substitution)
        """
        if self._ins_del_sub_cor is None:
            found_to_label = {item[0]: item[1]
                              for item in self.label_assignment}
            self._ins_del_sub_cor = {
                file: get_ins_del_cor_sub(
                    res_ref,
                    self.ins_del_valid[file],
                    found_to_label
                )
                for file, res_ref in self.res_ref.items()
            }

        return self._ins_del_sub_cor

    @property
    def unique_ins_del_sub_cor(self):
        """ return insertions, deletions, substitutions and correct pairs

        :return: list with indicators for each pair consisting of 'C' (correct),
                 'I' (insertion), 'D' (deletion) and 'S' (substitution)
        """
        if self._unique_ins_del_sub_cor is None:
            found_to_label = {item[0]: item[1]
                              for item in self.unique_label_assignment}
            self._unique_ins_del_sub_cor = {
                file: get_ins_del_cor_sub(
                    res_ref,
                    self.ins_del_valid[file],
                    found_to_label
                )
                for file, res_ref in self.res_ref.items()
            }

        return self._unique_ins_del_sub_cor

    @property
    def ins_del_sub_cor_p_r_f_sum(self):
        """ get sum of insertions, deletions, substitutions and correct pairs
            as well as precision, recall and f-score

        :return: correct, substitutions, deletions, insertions
                 precision, recall, f-score
        """
        if self._ins_del_sub_cor_p_r_f_sum is None:
            self._ins_del_sub_cor_p_r_f_sum = get_ins_del_sub_cor_p_r_f_sum(
                self.ins_del_sub_cor
            )

        return self._ins_del_sub_cor_p_r_f_sum

    @property
    def unique_ins_del_sub_cor_p_r_f_sum(self):
        """ get sum of insertions, deletions, substitutions and correct pairs
            as well as precision, recall and f-score

        :return: correct, substitutions, deletions, insertions
                 precision, recall, f-score
        """
        if self._unique_ins_del_sub_cor_p_r_f_sum is None:
            self._unique_ins_del_sub_cor_p_r_f_sum =\
                get_ins_del_sub_cor_p_r_f_sum(self.unique_ins_del_sub_cor)

        return self._unique_ins_del_sub_cor_p_r_f_sum


def read_ctm(file, pos=(0, 1, 2, 3), has_duration=False, file_transfrom=None,
             blacklist=(), add_bool=False, offset_from_filename=None,
             file_segments=None):
    """ read a ctm file

    :param file: ctm file to read from
    :param pos: 4 element tuple with positions of (filename, word, start, end)
    :param has_duration: ctm uses duration instead of end time
    :param file_transfrom: transform function to modify filename
    :param blacklist: blacklist of words to skip when reading
    :param add_bool: add boolen to ctm transcription
    :param offset_from_filename: function to derive time offset from filename
    :param file_segments: tuples containing (filename, start, end) of segments
                          ctm will be created with 'filename_start_end' as key
    :return: dict with transcription and timings per file

    Example for ctm with add_bool=True
    reference_ctm['c1lc021h'] = \
        [('MR.', 0.91, 1.26, True),
         ('MAUCHER', 1.26, 1.6, True)]
    """

    ctm = dict()
    with open(file, 'r') as fp:
        for line in fp:
            split_line = line.split()
            filename = split_line[pos[0]]
            word = split_line[pos[1]]
            start = float(split_line[pos[2]])
            end = float(split_line[pos[3]])

            if has_duration:
                end += start

            if offset_from_filename is not None:
                offset = offset_from_filename(filename)
                start += offset
                end += offset

            if file_transfrom is not None:
                filename = file_transfrom(filename)

            if word not in blacklist:
                if add_bool:
                    entry = (word, start, end, False)
                else:
                    entry = (word, start, end)

                if filename in ctm:
                    ctm[filename].append(entry)
                else:
                    ctm[filename] = [entry]

    ctm = {filename: sorted(entry, key=operator.itemgetter(1))
            for filename, entry in ctm.items()}

    if file_segments is not None:
        segments_ctm = dict()
        for file, start, end in file_segments:
            filename = '{}_{:06d}-{:06d}'.format(file,
                                                 int(round(start*1000)),
                                                 int(round(end*1000)))
            segments_ctm[filename] = [entry for entry in ctm[file]
                                      if entry[1] >= start and entry[2] <= end]

        return segments_ctm

    return ctm


def get_overlap(word1, word2):
    """ calculate overlap of two words, will be negative if there is no overlap

    :param word1: tuple (word, start, end)
    :param word2: tuple (word, start, end)
    :return: overlap of words
    """
    return min(word1[2], word2[2]) - max(word1[1], word2[1])


def overlap_to_cost(D):
    """
    Transform overlap to costmatrix : D = max(D) - D, in place operation
    :param D: distance matrix
    """
    D -= np.max(D)
    D *= -1


def get_ins_del_valid(res_ref):
    """
    Get insertions, deletions and valid pairs. A pair is marked as insertion,
    if a reference labelis assigned to multiple results. A pair is marked as
    deletion, if a result is assigned to multiple references
    :param res_ref: ctm with reference added to each entry
    :return: list with indicators for each pair consisting of 'V' (valid),
             'I' insertion and 'D' (deletion)
    """
    ins_del_valid = ['V'] * len(res_ref)

    res_prev = None
    res_max_overlap_duplicate = 0
    res_max_overlap_duplicate_idx = -1

    ref_prev = None
    ref_max_overlap_duplicate = 0
    ref_max_overlap_duplicate_idx = -1

    for idx, (res, overlap, ref) in enumerate(res_ref):
        if res != res_prev:
            res_max_overlap_duplicate = overlap
            res_max_overlap_duplicate_idx = idx
        else:
            if overlap > res_max_overlap_duplicate:
                assert ins_del_valid[res_max_overlap_duplicate_idx] == 'V'
                ins_del_valid[res_max_overlap_duplicate_idx] = 'D'
                res_max_overlap_duplicate_idx = idx
                res_max_overlap_duplicate = overlap
            else:
                assert ins_del_valid[idx] == 'V'
                ins_del_valid[idx] = 'D'

        res_prev = res

        if ref != ref_prev:
            ref_max_overlap_duplicate = overlap
            ref_max_overlap_duplicate_idx = idx
        else:
            if overlap > ref_max_overlap_duplicate:
                assert ins_del_valid[ref_max_overlap_duplicate_idx] == 'V'
                ins_del_valid[ref_max_overlap_duplicate_idx] = 'I'
                ref_max_overlap_duplicate_idx = idx
                ref_max_overlap_duplicate = overlap
            else:
                assert ins_del_valid[idx] == 'V'
                ins_del_valid[idx] = 'I'

        ref_prev = ref

    return ins_del_valid


def allign_res_ref(res, ref):
    """
    allign reference to result transcription
    :param res: list of resulting words
    :param ref: list of reference words
    :return: list with result, corresponding reference words and overlap:

    Example:
        [[('MIHSTAH', 0.95, 1.23), 0.27, ('MR.', 0.92, 1.22)],
         [('JHEYKAHBZ', 1.23, 1.74), 0.51, ('JACOBS', 1.22, 1.75)]]
    """
    c_min, d_pair, c_acc, path = dtw(res, ref, get_overlap, overlap_to_cost)
    return [[res[idx1], d_pair[idx1][idx2], ref[idx2]]
            for idx1, idx2 in zip(path[0], path[1])]


def get_conf_mat(res_ref, ins_del_valid):
    """ get confusion matrix from result + reference ctm. The confusion matrix
    is stored as a dictionary of dictionaries where the first dictionary is the
    discovered class and the second dictionary the corresponding labels with
    corresponding occurance count and accumulated overlap

    Example:
    conf_mat = \
        'AOTAH': {'OTHER': (1, 0.20), 'TO': (1, 0.14)},
        'GIHZ': {'GIVES': (1, 0.21), 'IS': (1, 0.18)},

    :param res_ref: result + reference ctm
    :param ins_del_valid: list indicating valid pairs
    :return: confusion matrix
    """
    conf_mat = dict()
    for file, sentence in res_ref.items():
        for idx, (res, overlap, ref) in enumerate(sentence):
            if ins_del_valid is None or ins_del_valid[file][idx] == 'V':
                if res[0] not in conf_mat:
                    conf_mat[res[0]] = dict()

                if ref[0] in conf_mat[res[0]]:
                    conf_mat[res[0]][ref[0]] =\
                        (conf_mat[res[0]][ref[0]][0] + 1,
                         conf_mat[res[0]][ref[0]][1] + overlap)
                else:
                    conf_mat[res[0]][ref[0]] = (1, overlap)

    return conf_mat


def get_ins_del_cor_sub(res_ref, ins_del_valid, found_to_label):
    """ mark wrong mappings in result and reference ctm. Correct mappings will
    be marked as 'C' in reference and result ctm. Results with a duplicate
    reference will be marked as insertions ('I'). Unused references will be
    marked as deletions, wrong mappings will be marked as substitutions.

    :param res_ref: ctm with result + reference pairs
    :param ins_del_valid: list with indicators for each pair consisting of
                          'V' (valid), 'I' insertion and 'D' (deletion)
    :param found_to_label: dict with found to label mapping pairs
                               {'hlo': 'Hello', ...}
    :return: list with indicators for each pair consisting of 'C' (correct),
             'I' (insertion), 'D' (deletion) and 'S' (substitution)
    """
    ins_del_cor_sub = list(ins_del_valid)

    # find insertions and substitutions in results
    for idx, (res, overlap, ref) in enumerate(res_ref):
        if ins_del_valid[idx] == 'V':
            if res[0] in found_to_label and found_to_label[res[0]] == ref[0]:
                ins_del_cor_sub[idx] = 'C'
            else:
                ins_del_cor_sub[idx] = 'S'

    return ins_del_cor_sub


def get_ins_del_sub_cor_p_r_f_sum(ins_del_cor_sub):
    """ get number of correct, substituted, deleted and inserted labels as well
    precision, recall, f-score

    :param ins_del_cor_sub: list with state markings
    :return: correct, substitutions, deletions, insertions
             precision, recall, f-score
    """
    correct = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    for sentence in ins_del_cor_sub.values():
        for state in sentence:
            if state == 'C':
                correct += 1
            if state == 'S':
                substitutions += 1
            if state == 'I':
                insertions += 1
            if state == 'D':
                deletions += 1

    precision = correct/(correct + substitutions + insertions)
    recall = correct/(correct + substitutions + deletions)
    fscore = 2*(precision*recall)/(precision+recall)

    return insertions, deletions, substitutions,\
           correct, precision, recall, fscore


def get_unique_label_assignment(found_to_label_assigments, max_type=None,
                                sort_type=None):
    """ Get best unique label assignment from label assignment (only keeps
        unique pairs with maximum overlap)

    :param found_to_label_assigments: list with found to label mapping pairs
                                      including overlap information
                                      (see get_label_assignment)
    :param max_type: criterion do decide for assignment:
                         'duration' (max accumulative overlap)
                         'count' (maximum accumulative count)
    :param sort_type: criterion for sorting the output:
                          'duration' (accumulative overlap)
                          'count' (accumulative count)
                          'count,duration' (count then accumulative overlap)
    :return: best unique label assignment list with pairs
             and reverse confusion matrix

    Example:
    assign = \
        [('DHAH', 'THE', (2517, 220.42)),
         ('IHN', 'IN', (807, 105.4)),
         ...]
    """
    max_key = lambda val: val[1][0]
    if max_type == 'duration':
        max_key = lambda val: val[1][1]

    sort_key = lambda item: item[2][0]
    if sort_type == 'duration':
        sort_key = lambda item: item[2][1]
    if sort_type == 'count,duration':
        sort_key = lambda item: (item[2][0], item[2][1])

    reverse_conf_mat = dict()
    for found, ref, overlap in found_to_label_assigments:
        if ref not in reverse_conf_mat:
            reverse_conf_mat[ref] = {found: overlap}
        else:
            reverse_conf_mat[ref][found] = overlap

    reverse_assign = ((ref,) + max(found.items(), key=max_key)
              for ref, found in reverse_conf_mat.items())

    best_unique_assign = ((found, ref, overlap)
                          for ref, found, overlap in reverse_assign)

    return sorted(best_unique_assign, key=sort_key, reverse=True),\
           reverse_conf_mat


def get_label_assignment(conf_mat, max_type='count', sort_type='count',
                         map_type='Best'):
    """ derive label assignment from confusion matrix

    :param conf_mat: confusion matrix
    :param max_type: criterion do decide for assignment:
                         'duration' (max accumulative overlap)
                         'count' (maximum accumulative count)
    :param sort_type: criterion for sorting the output:
                          'duration' (accumulative overlap)
                          'count' (accumulative count)
                          'count,duration' (count then accumulative overlap)
    :param map_mode: mapping mode:
                         'Best' (map found label to best overlapping reference
                                 label, duplicate mappings allowed)
                         'BestUnique' (map found label to best overlapping
                                       reference, duplicate mappins removed and
                                       only the best unique mapping is kept)
    :return: label assignment list with pairs

    Example:
    assign = \
        [('DHAH', 'THE', (2517, 220.42)),
         ('IHN', 'IN', (807, 105.4)),
         ...]
    """
    max_key = lambda val: val[1][0]
    if max_type == 'duration':
        max_key = lambda val: val[1][1]

    sort_key = lambda item: item[2][0]
    if sort_type == 'duration':
        sort_key = lambda item: item[2][1]
    if sort_type == 'count,duration':
        sort_key = lambda item: (item[2][0], item[2][1])

    assign = ((found,) + max(ref.items(), key=max_key)
              for found, ref in conf_mat.items())

    if map_type == 'BestUnique':
        return get_unique_label_assignment(assign, max_type, sort_type)[0]

    return sorted(assign, key=sort_key, reverse=True)


def get_max_overlap_sequence(word_entry, phoneme_entries):
    """ find maximum overlapping sequence for word entry in phoneme entries

    :param word_entry: word tuple: (word, start, end)
    :param phoneme_entries: list of phoneme tuples [(phoneme, start, end), ...]
    :return: sequence of phoneme tuples corresponding to word entry
    """

    word, word_start, word_end = word_entry
    sequence = tuple((phoneme, phoneme_start, phoneme_end)
                     for phoneme, phoneme_start, phoneme_end in phoneme_entries
                     if min(word_end, phoneme_end) -
                     max(word_start, phoneme_start)
                     >= 0.5*(phoneme_end - phoneme_start))
    return sequence


def add_max_overlap_sequences(word_entries, phoneme_entries):
    """ add maximum overlapping sequences to given word entries from
     phoneme entries

    :param word_entries: list of word tuples [(word, start, end), ...]
    :param phoneme_entries: list of phoneme tuples [(phoneme, start, end), ...]
    :return: tuple((word, sequence), ...)
    """
    return tuple((word_entry, get_max_overlap_sequence(word_entry,
                                                       phoneme_entries))
                for word_entry in word_entries)


def get_max_overlap_sequences_ctm(ctm_word, ctm_phoneme):
    """ Given a word ctm and a phoneme ctm, return a ctm with phoneme
    sequences added

    :param ctm_word: word ctm (dictionary with  list of word tuples)
    :param ctm_phoneme: phoneme ctm (dictionary with list of phoneme tuples)
    :return: ctm with word and corresponding phoneme sequences
    """
    return {id: add_max_overlap_sequences(ctm_word[id], ctm_phoneme[id])
            for id in ctm_word if id in ctm_phoneme}


def build_lexicon(word_pronunciations):
    """ build a lexicon from a ctm with word and phoneme sequences

    :param word_pronunciations: ctm with word and corresponding phoneme sequences
    :return: lexicon with sequences and correcponding occurance counts
    """
    lexicon = dict()
    for words in word_pronunciations.values():
        for word, pronunciation in words:
            sequence = tuple(char[0] for char in pronunciation)
            if word[0] in lexicon:
                if sequence in lexicon[word[0]]:
                    lexicon[word[0]][sequence] += 1
                else:
                    lexicon[word[0]][sequence] = 1
            else:
                lexicon[word[0]] = {sequence: 1}
    return lexicon


def get_sorted_lexicon(lexicon):
    """ create a sorted lexicon, sorted by number of occurances of words and
    pronunciations sorted by number of occurance

    :param lexicon: lexicon with sequences and correcponding occurance counts
    :return: sorted tuple of word and correcponding pronunciations
    """

    sort_key = lambda item: sum(item[1].values())

    return tuple((word, tuple(sorted(pronunciations.items(),
                                     key=operator.itemgetter(1),
                                     reverse=True)))
                 for word, pronunciations in sorted(lexicon.items(),
                                                    key=sort_key,
                                                    reverse=True))


def write_dict_file(lexicon, file, mincount=0):
    """ write a dictionary file from discovered pronunciations with entries:
    word pronuciation count

    :param lexicon: sorted lexicon
    :param file: ouput file
    :param mincount: minimum occurance count for output
    """
    if isinstance(lexicon, dict):
        lexicon = tuple(lexicon.items())

    with open(file, 'w') as dict_file:
        for word, pronunciations in lexicon:
            if isinstance(pronunciations, dict):
                pronunciations = tuple(pronunciations.items())

            for pronunciation, count in pronunciations:
                if count >= mincount:
                    dict_file.write(
                        '{}\t{}\t{}\n'.format(word,
                                              ' '.join(pronunciation),
                                              count))


def prune_ctm(ctm, min_num_char, min_duration):
    """ prune cmt and remove words below minimal character count and duration

    :param ctm: cmt dictionary
    :param min_num_char: minimum numebr of chars to keep
    :param min_duration: minimum duration to keep
    :return: pruned cmt dictionary
    """
    return {id: [word_entry for word_entry in word_entries
                 if len(word_entry[0]) >= min_num_char
                 and word_entry[2] >= min_duration]
            for id, word_entries in ctm.items()}

def write_clusters(ctm, out_filename, file_transfrom=None):
    """  Write cluster file for evaluation with https://github.com/bootphon/tde

    :param ctm: ctm file
    :param out_filename: output filename
    """
    clusters = dict()
    for file, sentence in ctm.items():
        for word in sentence:
            if word[0] not in clusters:
                clusters[word[0]] = list()

            if file_transfrom is not None:
                clusters[word[0]].append((file_transfrom(file), ) + word[1:])
            else:
                clusters[word[0]].append((file, ) + word[1:])

    with open(out_filename, 'w') as fid:
        for idx, words in enumerate(clusters.values()):
            fid.write('Class {}\n'.format(idx))
            for word in words:
                fid.write('{} {} {}\n'.format(*word))

            fid.write('\n')