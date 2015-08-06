import numpy
import json

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
          '........string\n')


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
            return data
        if isinstance(data, dict):
            return _build_dict(data)
        if isinstance(data, float):
            return data
        if isinstance(data, int):
            return data
        if isinstance(data, numpy.ndarray):
            return data.tolist()
        if isinstance(data, str):
            return data
        else:
            return 'Datatype not supported'

    def _build_dict(data):
        return {key: _filter(val) for key, val in data.items()}

    json.dump(_build_dict(dict_data), fid, sort_keys=True, indent=2)