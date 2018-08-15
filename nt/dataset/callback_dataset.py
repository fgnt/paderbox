import random
from nt.utils.deprecated import deprecated


@deprecated
def _identity_callback(data, **kwargs):
    return data


class UtteranceCallbackDataset():
    """ A template for a callback fetcher.

    """
    @deprecated
    def __init__(self, transformation_callback=None,
                 transformation_kwargs=None):

        if transformation_callback is None:
            self.callback = _identity_callback
        else:
            self.callback = transformation_callback
        if transformation_kwargs is not None:
            self.callback_kwargs = transformation_kwargs
        else:
            self.callback_kwargs = dict()

        self.utterances = self._get_utterance_list()

    @deprecated
    def __getitem__(self, index):
        """
        Returns an example or a sequence of examples.
        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            ret = []
            while current < stop and step > 0 or current > stop and step < 0:
                ret.append(self.get_example(current))
                current += step
            return ret
        elif isinstance(index, str):
            try:
                utt_idx = self.utterances.index(index)
            except ValueError:
                raise KeyError('{} is not a valid database key'.format(index))
            return self._get_utterance_for_utt(
                index, utt_idx
            )
        else:
            return self.get_example(index)

    @deprecated
    def __contains__(self, key):
        if isinstance(key, str):
            return key in self.utterances
        else:
            raise TypeError(type(key), key, 'Expect str')

    @deprecated
    def _get_utterance_list(self):
        """ Returns a list with utterance ids

        :return: List with utterance ids
        """
        raise NotImplementedError

    @deprecated
    def _read_utterance(self, utt):
        """ Reads the (untransformed) data for utterance `utt`

        :param utt:
        :return: dict with data
        """
        raise NotImplementedError

    @deprecated
    def __len__(self):
        return len(self.utterances)

    @deprecated
    def get_example(self, i):
        return self._get_utterance_for_idx(i)

    @deprecated
    def _get_utterance_for_idx(self, i):
        return self._get_utterance_for_utt(self.utterances[i], i)

    @deprecated
    def _get_utterance_for_utt(self, utt, utt_idx):
        """

        :param utt: Name or id of the utterance
        :param utt_idx: Index of the utterance in the utterance list
        :return: The utterance containing data and meta information
        """

        data = self._read_utterance(utt)
        data['utt_id'] = utt
        data['utt_idx'] = utt_idx
        data = self.callback(data, **self.callback_kwargs)
        return data

    @deprecated
    def shuffle_utterances(self):
        """ Shuffles utterance list again.

        This can be useful, if you want to have to identical fetchers yield
        different utterances at a time, i.e. for source separation.
        """
        random.shuffle(self.utterances)
