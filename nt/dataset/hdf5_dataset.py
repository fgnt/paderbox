import os

import h5py
import numpy as np
from natsort import natsorted
from nt.dataset.callback_dataset import UtteranceCallbackDataset
from nt.utils.deprecated import deprecated


class UtteranceHDF5Dataset(UtteranceCallbackDataset):
    """
    HDF5 dataset ready for parallel usage with MultiprocessIterator.
    """
    def __init__(
            self, hdf5_file, path, data_list,
            transformation_callback=None, transformation_kwargs=None
    ):
        self.hdf5_file = os.path.realpath(os.path.expanduser(hdf5_file))
        self.path = path
        self.data_list = data_list

        super().__init__(
            transformation_callback=transformation_callback,
            transformation_kwargs=transformation_kwargs
        )

    @deprecated
    @property
    def data(self):
        """
        Although, it may seem redundant to open the file every time an utterance
        is read, it is advisable, since the MultiprocessIterator starts a worker
        for every reading operation.
        """
        db = h5py.File(self.hdf5_file, 'r')
        for subtree in self.path.split('/'):
            db = db[subtree]
        return db

    @deprecated
    def _get_utterance_list(self):
        return natsorted(list(self.data.keys()))

    @deprecated
    def _read_utterance(self, utt):
        ret_data = dict()
        for key in self.data_list:
            ret_data[key] = np.asarray(self.data[utt][key])
        return ret_data
