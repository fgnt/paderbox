__author__ = 'walter'

import operator


def read_ctm(file, pos=(0, 1, 2, 3), has_duration=False, file_transfrom=None,
             blacklist=(), add_bool=False):
    """ read a ctm file

    :param file: ctm file to read from
    :param pos: 4 element tuple with positions of (filename, word, start, end)
    :param has_duration: ctm uses duration instead of end time
    :param file_transfrom: transform function to modify filename
    :param blacklist: blacklist of words to skip when reading
    :param add_bool: add boolen to ctm transcription
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

            if file_transfrom is not None:
                filename = file_transfrom(filename)

            if has_duration:
                end += start

            if word not in blacklist:
                if add_bool:
                    entry = (word, start, end, False)
                else:
                    entry = (word, start, end)

                if filename in ctm:
                    ctm[filename].append(entry)
                else:
                    ctm[filename] = [entry]
    return ctm


def get_overlap(word1, word2):
    """ calculate overlap of two words, will be negative if there is no overlap

    :param word1: tuple (word, start, end)
    :param word2: tuple (word, start, end)
    :return: overlap of words
    """
    return min(word1[2], word2[2]) - max(word1[1], word2[1])


def get_max_overlap_ref_for_word_entry(found_entry, ref_entries):
    """ find maximum overlapping reference for word entry in reference entries
    This function also modifies ref_entries and sets the boolean to true, if it
    is present

    :param found_entry: tuple: (word, start, end)
    :param ref_entries: list of referrnce tuples [(ref, start, end), ...]
    :return: reference tuple corresponding to word entry
    """

    max_key = lambda item: get_overlap(found_entry, item[1])
    idx, ref_entry = max(enumerate(ref_entries), key=max_key)

    # TODO: Check if overlap is negative --> insertion
    if len(ref_entry) > 3:
        ref_entries[idx] = ref_entry[0:3] + (True,)

    return ref_entry


def add_max_overlap_ref_to_word_entries(word_entries, ref_entries):
    """ add maximum overlapping reference to word entries from reference entries
    This will modify the reference as well to indicate if a reference label was
    found (if boolen is present in the reference tuple)

    :param word_entries: list of word tuples [(word, start, end), ...]
    :param ref_entries: list of referrnce tuples [(ref, start, end), ...]
    :return: list of word entries with corresponding references

    Example for word_entries:
    word_entries = \
        [(('MIHSTAH', 0.96, 1.26), ('MR.', 0.91, 1.26, True)),
         (('MAYKUHD', 1.26, 1.64), ('MAUCHER', 1.26, 1.6, True))]
    """
    return [(word_entry, get_max_overlap_ref_for_word_entry(word_entry,
                                                            ref_entries))
            for word_entry in word_entries]


def mark_longest_duplicate_labels(word_entries):
    """ mark longest match in duplicate reference labels as false and the
    remaining ones as true. This function will only consider duplicates, if one
    of the reference label has already been marked as a duplicate.

    :param word_entries: list of word entries
    """
    max_key = lambda item: get_overlap(item[1][0], item[1][1])
    for idx1, pair1 in enumerate(word_entries):
        if pair1[1][3]:
            duplicates = [(idx2, pair2)
                          for idx2, pair2 in enumerate(word_entries)
                          if pair1[1][:3] == pair2[1][:3]]
            idx_max_overlap, _ = max(duplicates, key=max_key)

            for idx, pair in duplicates:
                duplicate = True
                if idx == idx_max_overlap:
                    # TODO: Check if overlap is negative --> insertion
                    duplicate = False
                word_entries[idx] = (pair[0], pair[1][:3] + (duplicate,))
    return word_entries


def add_max_overlap_ref_to_ctm(ctm_word, ctm_ref):
    """ Given a word ctm and a reference ctm, add reference words to word ctm.
    This also modifies the reference ctm to indicate if a refernce
    label was found (if boolen is present in the reference tuple)

    :param ctm_word: word ctm (dictionary with  list of word tuples)
    :param ctm_ref: reference ctm (dictionary with list of refernce tuples)
    :return modified word ctm

    Example for modified result_ctm:
    result_ctm['c1lc021h'] = \
        [(('MIHSTAH', 0.96, 1.26), ('MR.', 0.91, 1.26, False)),
         (('MAYKUHD', 1.26, 1.64), ('MAUCHER', 1.26, 1.6, False))]
    """
    return {id: mark_longest_duplicate_labels(
            add_max_overlap_ref_to_word_entries(ctm_word[id], ctm_ref[id]))
            for id in ctm_word if id in ctm_ref}


def get_confusion_matrix(result_ctm):
    """ get confusion matrix from result ctm. The confusion matrix is stored as
    a dictionary of dictionaries where the first dictionary is the discovered
    class and the second dictionary the corresponding labels with corresponding
    occurance count and accumulated overlap

    Example:
    conf_mat = \
        'AOTAH': {'OTHER': (1, 0.20), 'TO': (1, 0.14)},
        'GIHZ': {'GIVES': (1, 0.21), 'IS': (1, 0.18)},

    :param result_ctm: result ctm
    :return: confusion matrix
    """
    conf_mat = dict()
    for sentence in result_ctm.values():
        for found, ref in sentence:
            if not ref[3]:
                if found[0] not in conf_mat:
                    conf_mat[found[0]] = dict()

                overlap = get_overlap(found, ref)

                if ref[0] in conf_mat[found[0]]:
                    conf_mat[found[0]][ref[0]] =\
                        (conf_mat[found[0]][ref[0]][0] + 1,
                         conf_mat[found[0]][ref[0]][1] + overlap)
                else:
                    conf_mat[found[0]][ref[0]] = (1, overlap)

    return conf_mat


def mark_wrong_mappings(result_ctm, found_to_label_assignements, ref_ctm):
    """ mark wrong mappings in result and reference ctm. Correct mappings will
    be marked as 'C' in reference and result ctm. Results with a duplicate
    reference will be marked as insertions ('I'). Unused references will be
    marked as deletions, wrong mappings will be marked as substitutions.

    :param result_ctm: ctm with result/label pairs and the use of duplicate
                       references marked
    :param found_to_label_assignements: list with found to label mapping pairs
                                        [('hlo', 'Hello'), ...]
    :param ref_ctm: reference ctm
    """
    found_to_label = {item[0]: item[1] for item in found_to_label_assignements}
    for id, sentence in result_ctm.items():
        # intialize reference label states to deleted
        for ref_idx, ref_label in enumerate(ref_ctm[id]):
            ref_ctm[id][ref_idx] = ref_label[:3] + ('D',)

        # find insertions and substitutions in results
        for idx, (found, ref) in enumerate(sentence):
            res = 'C'
            if ref[3]:
                res = 'I'
            elif (found[0] not in found_to_label) or\
                    (found_to_label[found[0]] != ref[0]):
                res = 'S'

            sentence[idx] = (found, ref[:3] + (res,))

            # mark reference as correct or substitutions, if result was not an
            # insertion
            if res != 'I':
                for ref_idx, ref_label in enumerate(ref_ctm[id]):
                    if ref_label[:3] == ref[:3]:
                        ref_ctm[id][ref_idx] = ref_label[:3] + (res,)
                        break


def get_unique_label_assigment(found_to_label_assigments, max_type=None,
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


def get_label_assignment(conf_mat, max_type=None, sort_type=None,
                         map_type=None):
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
        return get_unique_label_assigment(assign, max_type, sort_type)[0]

    return sorted(assign, key=sort_key, reverse=True)


def get_corr_sub_del_ins(result_ctm, ref_ctm):
    """ get number of correct, substituted, deleted and inserted labels
    See mark_wrong_mappings to create needed ctms

    :param result_ctm: result ctm with marked wrong mappings
    :param ref_ctm: reference ctm with marked wron mappeing
    :return: correct, substitutions, deletions, insertions
    """
    correct = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    for sentence in result_ctm.values():
        for _, (_, _, _, state) in sentence:
            if state == 'C':
                correct += 1
            if state == 'S':
                substitutions += 1
            if state == 'I':
                insertions += 1

    for sentence in ref_ctm.values():
        for _, _, _, state in sentence:
            if state == 'D':
                deletions += 1

    return correct, substitutions, deletions, insertions


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