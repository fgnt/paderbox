import json
import os


def load_json(*path_parts):
    """ Loads a json file and returns it as a dict

    :param path_parts: Json file name and possible parts of a path
    :return: dict with contents of the json file
    """

    path = os.path.join(*path_parts)

    if not path.endswith('.json'):
        path += '.json'

    with open(path) as fid:
        return json.load(fid)


def print_template():
    """ Prints the template used for the json file

    :return:
    """
    print('<root>\n'
          '..<train>\n'
          '....<flists>\n'
          '......<file_type> (z.B. wav)\n'
          '..........<scenario> (z.B. tr05_simu, tr05_real...)\n'
          '............<utterance_id>\n'
          '..............<observed>\n'
          '................<A>\n'
          '..................path\n'
          '................<B>\n'
          '..................path\n'
          '..............<image>\n'
          '................<A>\n'
          '..................path\n'
          '................<B>\n'
          '..................path\n'
          '..............<source>\n'
          '................path\n'
          '\n'
          '..<dev>\n'
          '..<test>\n'
          '..<orth>\n'
          '....<word>\n'
          '......<utterance_id>\n'
          '....<phoneme>\n'
          '......<utterance_id>\n'
          '........string\n'
          '..<flists>\n'
          '....Flist_1\n'
          '....Flist_2\n')


def print_old_template():
    """ Prints the template used for the json file

    :return:
    """
    print('<root>\n'
          '..<step_name>\n'
          '....<log>\n'
          '......list of strings\n'
          '....<config>\n'
          '......dict\n'
          '....<git_hash>\n'
          '......string\n'
          '....<date>\n'
          '......string\n'
          '....<comment>\n'
          '......string\n'
          '..<train>\n'
          '....<step_name>\n'
          '......Database / Feature extraction:\n'
          '......<flists>\n'
          '........<file_type> (z.B. wav)\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<observed>\n'
          '..................<A>\n'
          '....................string\n'
          '..................<B>\n'
          '....................string\n'
          '................<image>\n'
          '..................<A>\n'
          '....................string\n'
          '..................<B>\n'
          '....................string\n'
          '................<source>\n'
          '..................string\n'
          '......Beamformer:\n'
          '......<flists>\n'
          '........<file_type> (z.B. wav)\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<observed>\n'
          '....................string\n'
          '......<scores>\n'
          '..........<channels>\n'
          '............<scenario>\n'
          '..............<utterance_id>\n'
          '................<score_type>\n'
          '....................0 -> float (Wert vorher)\n'
          '....................1 -> float (Wert nachher)\n'
          '\n'
          '..<dev>\n'
          '..<test>\n'
          '..<orth>\n'
          '....<utterance_id>\n'
          '......string\n'
          '..<flists>\n'
          '....Flist_1\n'
          '....Flist_2\n')


def traverse_to_dict(data, path, delimiter='/'):
    """ Returns the dictionary at the end of the path defined by `path`

    :param data: A dict with the contents of the json file
    :param path: A string defining the path with or without
        leading and trailing slashes
    :param delimiter: The delimiter to convert the string to a list
    :return: dict at the end of the path
    """

    path = path.strip('/').split(delimiter)
    cur_dict = data[path[0]]
    for next_level in path[1:]:
        try:
            cur_dict = cur_dict[next_level]
        except KeyError as e:
            print('Error: {k} not found. Possible keys are {keys}'
                  .format(k=next_level, keys=cur_dict.keys()))
            raise e
    return cur_dict


def get_available_channels(data):
    """ Returns all available channels in the format *type/channel_no*

    :param data: A dictionary with ids as keys and file lists as values
    :type data: dict

    :return: A list of available channels
    """

    utt = list(data.keys())[0]
    channels = list()
    for src in data[utt]:
        if type(data[utt][src]) is dict:
            for ch in data[utt][src]:
                channels.append('{src}/{ch}'.format(src=src, ch=ch))
        else:
            channels.append(src)
    return channels


def get_flist_for_channel(flist, ch):
    """ Returns a flist containing only the files for a specific channel

    :param flist: A dict representing a file list
    :param ch: The channel to get

    :return: A dict with the ids and the files for the specific channel
    """

    assert ch in get_available_channels(flist), \
        'Could not find channel {ch}. Available channels are {chs}' \
        .format(ch=ch, chs=get_available_channels(flist))

    ret_flist = dict()
    for utt in flist:
        val = flist[utt]
        for branch in ch.split('/'):
            if branch in val:
                val = val[branch]
            else:
                return []
        ret_flist[utt] = val

    assert len(ret_flist) > 0, \
        'Could not find any files for channel {c}'.format(c=str(ch))
    return ret_flist


def safe_dump(dict_data, fid):
    """ Writes a dict to a json, ignoring all type which cannot be serialized

    :param fid:
    :param dict:
    :return:
    """

    def _filter(data):
        if isinstance(data, list):
            return [_filter(d) for d in data]
        if isinstance(data, dict):
            return _build_dict(data)
        if isinstance(data, float):
            return data
        if isinstance(data, int):
            return data
        if hasattr(data, 'tolist') and callable(data.tolist):
            return data.tolist()
        if isinstance(data, str):
            return data
        if isinstance(data, tuple):
            return [_filter(d) for d in data]
        else:
            return 'Datatype {} not supported'.format(type(data))

    def _build_dict(data):
        return {key: _filter(val) for key, val in data.items()}

    json.dump(_build_dict(dict_data), fid, sort_keys=True, indent=2)


def add_flist(flist, progress_json, scenario, stage='train',
              file_type='wav', channel_type='observed', channel='CH1'):
    """ Adds a file list to the current progress_json object

    Example::

    ....<flists>
    ......<file_type> (z.B. wav)
    ........<scenario> (z.B. tr05_simu, tr05_real...)
    ..........<utterance_id>
    ............<observed>
    ..............<A>

    :param flist: A dictionary acting as a file list
    :param progress_json: The current json object
    :param scenario: Name for the file list
    :param stage: [train, dev, test]
    :param file_type: Type of the referenced files. e.g. wav, mfcc, ...
    :return:
    """

    def _get_next_dict(cur_dict, key):
        try:
            return cur_dict[key]
        except KeyError:
            cur_dict[key] = dict()
            return cur_dict[key]

    cur_dict = progress_json[stage]
    flists_dict = _get_next_dict(cur_dict, 'flists')
    file_type_dict = _get_next_dict(flists_dict, file_type)
    scenario_dict = _get_next_dict(file_type_dict, scenario)

    for utt_id in flist:
        utt_id_dict = _get_next_dict(scenario_dict, utt_id)
        channel_type_dict = _get_next_dict(utt_id_dict, channel_type)
        channel_type_dict[channel] = flist[utt_id]


def combine_flists(data, flist_1_path, flist_2_path, flist_path,
                   postfix_1='', postfix_2='', delimiter='/'):
    """ Combines two file lists into a new file list ``flist_name``

    The new file list will only have those channels, which are present in both
    file lists.

    :param flist_1_path: Path to the first file list
    :param flist_2: Path to the second file list
    :param flist_name: Path to the new file list
    """

    flist_1 = traverse_to_dict(data, flist_1_path, delimiter)
    flist_2 = traverse_to_dict(data, flist_2_path, delimiter)

    if postfix_1 == '' and postfix_2 == '':
        assert len(set(list(flist_1.keys())+list(flist_2.keys()))) \
            == len(flist_1) + len(flist_2), \
            'The ids in the file lists must be unique.'

    channels_flist_1 = get_available_channels(traverse_to_dict(
        data, flist_1_path, delimiter
    ))
    channels_flist_2 = get_available_channels(traverse_to_dict(
        data, flist_2_path, delimiter
    ))

    common_channels = set((ch.split('/')[0]
                           for ch in channels_flist_1
                           if ch in channels_flist_2))

    new_flist = dict()
    for flist, postfix in zip([flist_1, flist_2], [postfix_1, postfix_2]):
        for id in flist.keys():
            new_id = id if len(postfix) == 0 else id + '_' + postfix
            new_flist[new_id] = dict()
            for ch in flist[id]:
                if ch in common_channels:
                    new_flist[new_id][ch] = flist[id][ch]

    flist_name = flist_path.split(delimiter)[-1]
    flist_parent_path = delimiter.join(flist_path.split(delimiter)[:-1])

    flist_parent = traverse_to_dict(data, flist_parent_path, delimiter)
    flist_parent[flist_name] = new_flist
