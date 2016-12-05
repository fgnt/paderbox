import unittest
import os.path
import numpy as np
from nt.io import audioread, audiowrite
from nt.speech_enhancement import wpe
import time
from nt import testing
from nt.io.data_dir import testing as testing_dir
from nt.io.data_dir import testing as data_dir
from nt.io.data_dir import DataDir


class TestWPEWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.settings_file_path = data_dir / 'speech_enhancement' / 'utils'
        self.audiofiles_path = data_dir / 'speech_enhancement' / 'data'
        self.sample_rate = 16000

    @testing.attr.matlab
    def test_dereverb_one_channel(self):
        input_file_paths =\
            {str(self.audiofiles_path / 'sample_ch1.wav'): 1, }
        self.process_dereverbing_framework(input_file_paths)
        time.sleep(2) # wait 2 seconds for audio files to pop up in file manager

        # check if audio files exist
        for utt, num_channels in input_file_paths.items():
            for cha in range(num_channels):
                self.assertTrue(
                    os.path.isfile(
                        utt.replace('ch1', 'ch'+str(cha+1)+'_derev')))

    @testing.attr.matlab
    def test_dereverb_eight_channels(self):
        input_file_paths =\
            {str(self.audiofiles_path / 'sample_ch1.wav'): 8, }
        self.process_dereverbing_framework(input_file_paths)
        time.sleep(2) # wait 2 seconds for audio files to pop up in file manager

        # check if audio files exist
        for utt, num_channels in input_file_paths.items():
            for cha in range(num_channels):
                utt_to_check = utt.replace('ch1', 'ch'+str(cha+1)+'_derev')
                # print('ch'+str(cha+1)+'_derev')
                self.assertTrue(os.path.isfile(utt_to_check))

    def process_dereverbing_framework(self, input_file_paths):
        """
        Require MATLAB
        """
        file_no = 0
        for utt, num_channels in input_file_paths.items():
            file_no += 1
            print("Processing file no. {0} ({1} file(s) to process in total)"
                  .format(file_no, len(input_file_paths)))
            noisy_audiosignals = np.ndarray(
                shape=[audioread.getparams(utt).nframes, num_channels],
                dtype=np.float32)

            for cha in range(num_channels):

                # erase old written audio files
                utt_to_be_written = utt.replace('ch1', 'ch'+str(cha+1)+'_derev')
                if os.path.isfile(utt_to_be_written):
                    os.remove(utt_to_be_written)

                # read microphone signals (each channel)
                print(" - Reading channel "+str(cha+1))
                utt_to_read = utt.replace('ch1', 'ch'+str(cha+1))
                signal = audioread.audioread(path=utt_to_read,
                                             sample_rate=self.sample_rate
                                             )
                if not noisy_audiosignals.shape[0] == len(signal):
                    raise Exception("Signal " + utt_to_read +
                                    " has a different size than other signals.")
                else:
                    noisy_audiosignals[:, cha] = signal

            # dereverb
            y = wpe.dereverb(self.settings_file_path, noisy_audiosignals)

            # write dereverbed signals (each channel)
            for cha in range(num_channels):
                utt_to_write = utt.replace('ch1', 'ch'+str(cha+1)+'_derev')
                print(" - Writing channel "+str(cha+1))
                audiowrite.audiowrite(y[:, cha],
                                      utt_to_write,
                                      self.sample_rate )
        print("Finished successfully.")


class TestMultichannelWPE(unittest.TestCase):

    def test_vectorized_dereverb(self):
        K = 10
        Delta = 1
        y = np.random.uniform(0, 1, (10, 3, 20))
        G_hat = np.random.uniform(0, 1, (10, K, 3, 3))
        ref = wpe._dereverberate(y, G_hat, K, Delta)
        vec = wpe._dereverberate_vectorized(y, G_hat, K, Delta)
        testing.assert_allclose(ref[:, :, K+Delta:], vec[:, :, K+Delta:],
                                rtol=1e-2)

    def test_multichannel_wpe_dimensions(self):
        settings = [
            (20, 3, 15, 1, 0),
            (20, 3, 15, 3, 0),
            (20, 3, 15, 3, 2),
        ]

        for setting in settings:
            L, N, T, K, Delta = setting

            Y = np.random.normal(size=(L, N, T)) + 1j * np.random.normal(size=(L, N, T))

            _ = wpe.multichannel_wpe(Y, K, Delta)


class TestGetCrazyMatrix(unittest.TestCase):
    def _generate_observation_matrix(self, L, N, T):
        Y = np.empty((L, N, T), dtype='<U6')
        for l in range(L):
            for n in range(N):
                for t in range(T):
                    Y[l, n, t] = 'l{}n{}t{}'.format(l, n, t)
        return Y

    def _check_alphabet_example(self, L, N, T, K, Delta):
        Y = self._generate_observation_matrix(L, N, T)
        def replace(x):
            if x == '':
                return '      '
            else:
                return x
        vector_replace = np.vectorize(replace)

        psi_bar = wpe._get_crazy_matrix(Y, K=K, Delta=Delta)
        psi_bar = vector_replace(psi_bar)

        def _get_fname(L, N, T, K, Delta):
            file = 'wpe_psi_bar_L{}_N{}_T{}_K{}_Delta{}.npy'.format(
                L, N, T, K, Delta
            )
            file = testing_dir / 'speech_enhancement' / 'data' / file
            return str(file)

        def _load(L, N, T, K, Delta):
            return np.load(_get_fname(L, N, T, K, Delta))

        assert np.array_equal(psi_bar, _load(L, N, T, K, Delta))

    def _get_crazy_matrix_loopy(self, Y, K, Delta):
        L, N, T = Y.shape
        dtype = Y.dtype
        psi_bar = np.zeros((L, N*N*K, N, T-Delta-K+1), dtype=dtype)
        for l in range(L):
            for n0 in range(N):
                for n1 in range(N):
                    for tau in range(Delta, Delta + K):
                        for t in range(Delta+K-1, T):
                            psi_bar[
                                l, N*N*(tau-Delta) + N*n0 + n1, n0, t-Delta-K+1
                            ] = Y[l, n1, t-tau]
        return psi_bar

    def test_alphabet_example1(self):
        self._check_alphabet_example(L=1, N=6, T=4, K=2, Delta=0)

    def test_alphabet_example2(self):
        self._check_alphabet_example(L=1, N=2, T=6, K=2, Delta=1)

    def test_compare_crazy_matrix_loopy_vectorized(self):
        L = 1
        N = 6
        T = 4
        K = 2
        Delta = 0
        Y = np.random.normal(size=(L, N, T)) + 1j * np.random.normal(size=(L, N, T))
        a = self._get_crazy_matrix_loopy(Y, K, Delta)
        b = wpe._get_crazy_matrix(Y, K, Delta)

        np.testing.assert_allclose(a, b)
