import collections
import concurrent.futures
import fnmatch
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
from nt.dataset.callback_dataset import UtteranceCallbackDataset

from nt.database.helper import *
from nt.io import load_json
from nt.io.audioread import audioread, read_nist_wsj


def get_matching_names(names, glob, regex):
    """
    Should this function moved to nt.utils? Rename something?

    This functions return a subset of the input strings which
        match to glob or regex.

    :param names: list or tuple of strings
    :param glob: Unix shell-style Regular Expressions.
        list of strings or string
    :param regex: Regular Expressions, perl-stype.
        list of strings or string
    :return: subset of names. list
    """
    if glob or regex:
        if not isinstance(regex, collections.Iterable):
            if regex:
                regex = regex,
            else:
                regex = list()

        if glob:
            if not isinstance(glob, collections.Iterable):
                glob = glob,
            regex += [fnmatch.translate(g) for g in glob]

        reobj = [re.compile(r) for r in regex]
        matched_names = [ch for ch in names if any(r.match(ch) for r in reobj)]

        assert len(matched_names) <= len(names)
        return matched_names
    return []


class UtteranceJsonCallbackDataset(UtteranceCallbackDataset):
    """ A callback fetcher which reads its data from a json file

    A callback fetcher reads data from different source (to be implemented in
    the actual class) and transforms it using a callback function. Afterwards,
    it might split this data into frames (see `frames` mode) or keep the whole
    utterance.

    The fetcher has two different modes:

    #. 'utterance': In this mode the data is read utterance-wise.
    #. 'frames': In this mode, a batch is compiled of single time frames
       mixed across multiple utterances. The fetcher buffers several
       utterances, concatenates them on the first dimension and then
       samples frames from this array. This has some implications:

        * The first dimension is assumed to be the time dimension for
          all requested arrays.
        * The data provider is not able to shuffle the frames anymore
          as not all frames are available at a given time. Shuffling
          should therefor be disabled for the data provider. The
          frames can still be shuffled by the fetcher though.

    A callback function takes the batch as a dictionary, can modify the
    signals and can add new dictionary entries. An example is given here:
    >>> import numpy as np
    >>> def augment(batch):
    >>>     shape = batch['X'].shape
    >>>     batch['N'] = np.random.normal(size=shape)
    >>>     return batch

    name, json_src, flist,
                 feature_channels=None, feature_channels_glob=None,
                 feature_channels_regex=None,
                 annotations=None, audio_start_key=None,
                 audio_end_key=None, context_length=None, nist_format=False,
                 sample_rate=16000, utt_to_skip=None,

    :param name: Name of the fetcher. This will correspond to the key of the
        dictionary output of the data provider.
    :param json_src: A path to the database json or a dict with the same
        structure
    :param flist: file list within the json holding the utterances
    :param feature_channels: Channels to load from the file list
    :param feature_channels_glob: Glob expression for the channels to load
    :param feature_channels_regex: Regular expression for the channels to load
    :param annotations: Path to possible annotations within the json
    :param audio_start_key: Name of the key within the annotations indicating
        the start of an utterance
    :param audio_end_key: Name of the key within the annotations indicating
        the start of an utterance
    :param context_length: Context to prepend to the utterance (needs
        annotations to work!)
    :param nist_format: Flag indicating if the audio is stored in nist format
    :param sample_rate: Target sample rate. If the audio has a different rate,
        it will be automatically resampled!
    :param utt_to_skip: List of utterances to skip
    :param sequence_features: If true, the data will have the form TxBxF. Note
        that this only affects the utterance mode. Sequential feature do not
        make sense for a frame-wise mode. This also only applies to data in
        the context list.
    :param left_context: Number of frames to concatenate from left context
        of current frame. Will pad with zeros if frames are not available.
    :param right_context:  Number of frames to concatenate from right context
        of current frame. Will pad with zeros if frames are not available.
    :param step: If configured in utterance mode and if context is added,
        this specifies the step-width of the context window.
    :param cnn_features: If true, the data will be transformed to a BxCxHxW
        format.
    :param deltas_as_channel: For CNN features, the deltas will be used as
        channels. This assumes that the deltas are present in the feature
        dimension of the (transformed) data. You also have to specify how
        many delta features are present using the `num_deltas` parameter.
    :param num_deltas: Has to be specified for a correct split of the feature
        dimension if the deltas should be a channel (see parameter above).
    :param shuffle_frames: Is ignored in utterance mode.
    :param buffer_size: Number of files loaded and used for shuffling.
        Is consumed entirely, before next files are loaded.
    :param callback: Callback function which takes a batch as
        a dictionary and possibly modifies the signals before a context is
        appended and before the signal is chopped into frames.
    :param callback_kwargs: Additional arguments passed to the augmentation
        callback.
    :param sample_rate: Resamples utterances to given value. None = native
        sample rate.
    """

    def __init__(self, json_src, flist, feature_channels, *,
                 transformation_callback=None,
                 transformation_kwargs=None, sample_rate=None):

        if isinstance(json_src, dict):
            self.src = json_src
        else:
            self.src = load_json(json_src)

        self.flist = traverse_to_dict(self.src, flist)
        if isinstance(feature_channels, str):
            feature_channels = [feature_channels]
        self.feature_channels = feature_channels
        self.sample_rate = sample_rate

        flist_parts = Path(flist).parts
        self.set = flist_parts[0]
        self.flist_name = flist_parts[-1]

        super().__init__(
            transformation_callback=transformation_callback,
            transformation_kwargs=transformation_kwargs
        )

    def _get_utterance_list(self):
        """ Returns a list with utterance ids

        :return: List with utterance ids
        """
        utt_id_lists = [get_flist_for_channel(self.flist, ch) for ch in
                        self.feature_channels]
        common_utt_ids = set.intersection(*map(set, utt_id_lists))

        # I decided to use `sorted` instead of `natsorted` here, because that
        # is the same sorting mechanism which is used in Kaldi.
        return sorted([utt_id for utt_id in common_utt_ids])

    def sort_by_file_size(self, reverse=False):
        """Sort the utterances by file size.

        When file size coincide, sort by utt id.

        Overwrites self.utterances accordingly.
        Much faster than sorting it by length.

        This does not work properly with the current way Chainer handles the
        end of an epoch. Currently, at the end of an epoch, when len(dataset)
        is not an integer multiple of batch_size, new examples from the next
        batch are mixed with the last batch. After a few iterations, the idea
        of this function is undermined.

        Args:
            reverse: False (Ascending), True (Descending)

        Returns:

        """

        def get_filesize(utt):
            channel = self.feature_channels[0]
            wav_file = get_channel_for_utt(self.flist, channel, utt)
            return os.path.getsize(wav_file)

        file_sizes = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_utt_id = {executor.submit(get_filesize, utt): utt for utt
                                in self.flist}
            for future in concurrent.futures.as_completed(future_to_utt_id):
                utt = future_to_utt_id[future]
                size = future.result()
                file_sizes[utt] = size

        keys = sorted(
            file_sizes,
            key=lambda key: (file_sizes[key], key),
            reverse=reverse
        )
        ordered_flist = OrderedDict()
        for k in keys:
            ordered_flist[k] = self.flist[k]
        self.flist = ordered_flist

        # The iterators access entries by order in self.utterances.
        self.utterances = list(self.flist.keys())

    def _read_audio(self, wav_file, utt):
        return audioread(wav_file, sample_rate=self.sample_rate)
        # return audioread(wav_file)

    def _read_utterance(self, utt):

        data_dict = dict()
        channels = self.feature_channels

        for channel in sorted(channels):
            wav_file = get_channel_for_utt(self.flist, channel, utt)
            data = self._read_audio(wav_file, utt)
            data = np.atleast_2d(data)
            ch_group = channel.split('/')[0]
            try:
                data_dict[ch_group] = np.concatenate(
                    [data_dict[ch_group], data], axis=0)
            except KeyError:
                data_dict[ch_group] = data

        assert len(data_dict) > 0, 'Did not load any audio data.' \
                                   'Maybe the channels do not exist?'

        return data_dict

    def sort(self, key='nsamples', reverse=False):
        """
        Sort `utterances` by a key given in the annotations of the json

        Args:
            key: Annotation to sort the utterances. Defaults to
                'nsamples'
            reverse: If True sort in descending order

        Returns:
            `utterances` sorted by `key`

        """

        annotations = traverse_to_dict(self.src, str(Path(self.set) /
                                                     'annotations' /
                                                     self.flist_name
                                                     )
                                       )
        try:  # Some annotations (e.g. `nsamples`) are stored in 'utterances'
            assert key in annotations[self.utterances[0]]
        except AssertionError:
            try:
                assert key in self.flist[self.utterances[0]]
            except AssertionError:
                raise ValueError('Key "{}" for dataset "{}" is not in json'.
                                 format(key, self.flist_name)
                                 )
            annotations = self.flist

        utt_to_key = dict()
        for utt in self.utterances:
            utt_to_key[utt] = annotations[utt][key]

        sorted_list = sorted(utt_to_key.items(),
                             key=lambda x: x[1], 
                             reverse=reverse)

        self.utterances = list(OrderedDict(sorted_list).keys())


class NistUtteranceJsonCallbackDataset(UtteranceJsonCallbackDataset):
    def _read_audio(self, wav_file, utt):
        return read_nist_wsj(wav_file)


class FlacUtteranceJsonCallbackDataset(UtteranceJsonCallbackDataset):
    def _read_audio(self, flac_file, utt):
        data, sample_rate = sf.read(flac_file, dtype=np.float32)
        return data


class ContextUtteranceJsonCallbackDataset(UtteranceJsonCallbackDataset):
    def __init__(self, json_src, flist, feature_channels, annotations,
                 audio_start_key=None,
                 audio_end_key=None, context_length=None,
                 transformation_callback=None,
                 transformation_kwargs=None, sample_rate=16000):

        if isinstance(json_src, dict):
            src = json_src
        else:
            src = load_json(json_src)

        if isinstance(annotations, dict):
            self.annotations = annotations
        else:
            self.annotations = traverse_to_dict(src, annotations)
        if not audio_start_key or not audio_end_key:
            raise ValueError('Please specify the start and end key for '
                             'the annotations.')
        self.start_key = audio_start_key
        self.end_key = audio_end_key
        self.context_length = context_length
        self.context_samples = 0

        super().__init__(json_src, flist, feature_channels,
                         transformation_callback=transformation_callback,
                         transformation_kwargs=transformation_kwargs,
                         sample_rate=sample_rate
                         )

    def _read_utterance(self, utt):

        data_dict = dict()
        start_context = max(0, (
            (self.annotations[utt][self.start_key] - self.context_length)))
        context_samples = \
            (self.annotations[utt][self.start_key] - start_context) * \
            self.sample_rate
        duration = self.annotations[utt][self.end_key] - start_context
        for channel in sorted(self.feature_channels):
            try:
                wav_file = get_channel_for_utt(self.flist, channel, utt)
                data = audioread(wav_file, start_context, duration,
                                 sample_rate=self.sample_rate)
                data = np.atleast_2d(data)
                ch_group = channel.split('/')[0]
                try:
                    data_dict[ch_group] = np.concatenate(
                        [data_dict[ch_group], data], axis=0)
                except KeyError:
                    data_dict[ch_group] = data
            except KeyError as e:
                raise e

        assert len(data_dict) > 0, 'Did not load any audio data. ' \
                                   'Maybe the channels do not exist?'

        data_dict['_context_samples'] = np.asarray(context_samples)

        return data_dict
