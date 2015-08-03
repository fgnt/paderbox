def print_template():
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
          '......string\n')


def traverse_to_dict(data, path, delimiter='/'):
    """ Returns the dictionary at the end of the path defined by `path`

    :param data: A dict with the contents of the json file
    :param path: A string defining the path
    :param delimiter: The delimiter to convert the string to a list
    :return: dict at the end of the path
    """

    path = path.split(delimiter)
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

    utt = data.keys().__next__()
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
