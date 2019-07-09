import unittest
from os import path

import numpy as np

import paderbox.testing as tc
from paderbox.io.data_dir import testing as testing_dir
from paderbox.io.audioread import audioread
from paderbox.speech_enhancement.beamformer import get_gev_vector
from paderbox.speech_enhancement.beamformer import get_lcmv_vector
from paderbox.speech_enhancement.beamformer import get_mvdr_vector
from paderbox.speech_enhancement.beamformer import get_pca_vector
from paderbox.speech_enhancement.beamformer import get_power_spectral_density_matrix
from paderbox.speech_enhancement.mask_module import biased_binary_mask, \
    wiener_like_mask
from paderbox.math.vector import vector_H_vector
from paderbox.utils.matlab import Mlab

# uncomment, if you want to test matlab functions
# matlab_test = unittest.skipUnless(True,'matlab-test')

def rand(*shape, data_type):
    if not shape:
        shape = (1,)
    elif isinstance(shape[0], tuple):
        shape = shape[0]

    def uniform(data_type_local):
        return np.random.uniform(-1, 1, shape).astype(data_type_local)

    if data_type in (np.float32, np.float64):
        return uniform(data_type)
    elif data_type is np.complex64:
        return uniform(np.float32) + 1j * uniform(np.float32)
    elif data_type is np.complex128:
        return uniform(np.float64) + 1j * uniform(np.float64)


class TestBeamformerMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        F : bins, number of frequencies
        T : time
        D : sensors, number of microphones
        K : sources, number of speakers
        :return:
        """
        datafile = testing_dir / 'speech_enhancement' / 'data' / 'beamformer.npz'
        datafile_multi_speaker = path.join(path.dirname(path.realpath(__file__)), 'data_multi_speaker.npz')

        self.mlab = Mlab()

        if not path.exists(datafile_multi_speaker):
            self.generate_source_file()

        with np.load(str(datafile)) as data:
            X = data['X']  # DxTxF
            Y = data['Y']
            N = data['N']
        ibm = biased_binary_mask(np.stack([X[4, :, :], N[4, :, :]]))
        self.Y_bf, self.X_bf, self.N_bf = Y.T.transpose(
            0, 2, 1), X.T.transpose(
            0, 2, 1), N.T.transpose(
            0, 2, 1)
        self.ibm_X_bf = ibm[0].T
        self.ibm_N_bf = ibm[1].T
        self.ibm_X_bf_th = np.maximum(self.ibm_X_bf, 1e-4)
        self.ibm_N_bf_th = np.maximum(self.ibm_N_bf, 1e-4)
        self.Phi_XX = get_power_spectral_density_matrix(self.Y_bf, self.ibm_X_bf_th)
        self.Phi_NN = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf_th)
        self.Phi_NN = self.Phi_NN + np.tile(1e-10 * np.eye(self.Phi_NN.shape[1]), (self.Phi_NN.shape[0], 1, 1))
        self.W_pca = get_pca_vector(self.Phi_XX)
        self.W_mvdr = get_mvdr_vector(self.W_pca, self.Phi_NN)
        self.W_gev = get_gev_vector(self.Phi_XX, self.Phi_NN)

        with np.load(datafile_multi_speaker) as data:
            X = data['X']  # K F D T
            Y = data['Y']  # F D T
            N = data['N']  # F D T
            self.data_multi_speaker = {'X': data['X'], 'Y': data['Y'], 'N': data['N']}
        masks = wiener_like_mask(np.concatenate([X, N[None, ...]]),
                                 sensor_axis=-2)
        X_mask, N_mask = np.split(masks, (X.shape[0],))
        # (K, F, T), (F, T)

        Phi_XX = get_power_spectral_density_matrix(Y, X_mask, source_dim=0)  # (K F D D)
        Phi_NN = get_power_spectral_density_matrix(Y, N_mask[0])  # (F D D)

        W_pca = get_pca_vector(Phi_XX)  # (K, F, D)
        W_mvdr = get_mvdr_vector(W_pca, Phi_NN)  # (K, F, D)
        W_gev = np.zeros_like(W_mvdr)
        W_gev[0, ...] = get_gev_vector(Phi_XX[0, ...], Phi_NN)
        W_gev[1, ...] = get_gev_vector(Phi_XX[1, ...], Phi_NN)

        W_lcmv = get_lcmv_vector(W_pca, [1, 0], Phi_NN)

    @staticmethod
    def generate_source_file():
        import paderbox.transform as transform
        from paderbox.speech_enhancement.noise.generator import NoiseGeneratorSpherical
        from paderbox.reverb.reverb_utils import generate_rir, convolve
        from paderbox.reverb.scenario import generate_sensor_positions, generate_source_positions_on_circle

        # ToDo: replace with Python funktions (current missing)
        D = 6
        K = 2
        SNR = 15
        soundDecayTime = 0.16
        samplingRate = 16000
        noisetype = np.random.randn
        secs = 2
        rirFilterLength = 2**14
        sensor_positions = generate_sensor_positions(number_of_sensors=D)[:,:D]
        sourceDoA = np.deg2rad([30, -30, 60, -60, 90, -90, 15, -15, 45, -45, 75, -75, 0])
        sourceDoA = sourceDoA[:K]
        source_positions = generate_source_positions_on_circle(azimuth_angles=sourceDoA)
        rir =  generate_rir(np.array([3,3,3]),source_positions, sensor_positions, soundDecayTime,
                sample_rate=samplingRate, filter_length=rirFilterLength, sensor_orientations=None,
                sensor_directivity=None, sound_velocity=343, algorithm=None
        )
        speaker_files = testing_dir / 'timit' / 'data'
        speakers = np.array(
            [audioread(str(speaker_files / f'sample_{num+1}.wav'),
                       expected_sample_rate=samplingRate)[0][:secs*samplingRate]
             for num in range(K)]
        )
        speakers = convolve(speakers, rir, truncate=True)
        generator = NoiseGeneratorSpherical(sensor_positions,
                                            sample_rate=samplingRate)

        noise = generator.get_noise_for_signal(
            np.sum(speakers, axis=0),  # sum above the speakers
            snr=SNR,
            rng_state=noisetype,
        )

        Speakers = transform.stft(speakers).transpose(1,2,3,0)
        Noise = transform.stft(noise)

        Y = np.sum(np.concatenate((Speakers, Noise[:, :, :, np.newaxis]), axis=3), axis=3).transpose(2, 0, 1).copy()
        X = Speakers.transpose(3, 2, 0, 1).copy()
        N = Noise.transpose(2, 0, 1).copy()

        datafile_multi_speaker = path.join(
            path.dirname(
                path.realpath(__file__)),
            'data_multi_speaker.npz')
        np.savez(datafile_multi_speaker, X=X, Y=Y, N=N)

    @tc.attr.matlab
    def test_compare_PSD_without_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.run_code('Phi = random.covariance(Y, [], 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @tc.attr.matlab
    def test_compare_PSD_with_mask(self):
        mlab = self.mlab.process
        mlab.set_variable('Y', self.Y_bf)
        mlab.set_variable(
            'ibm', self.ibm_N_bf[
                :, np.newaxis, :].astype(
                np.float))
        mlab.run_code('Phi = random.covariance(Y, ibm, 3, 2);')
        Phi_matlab = mlab.get_variable('Phi')
        Phi = get_power_spectral_density_matrix(self.Y_bf, self.ibm_N_bf)
        tc.assert_allclose(Phi, Phi_matlab, atol=1e-4)

    @tc.attr.matlab
    def test_compare_PCA_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.run_code("W = bss.beamformer.pca('cleanObservationMatrix', Phi_XX);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_pca)

    def test_mvdr_beamformer(self):
        tc.assert_allclose(vector_H_vector(self.W_pca, self.W_mvdr), 1)

    @tc.attr.matlab
    def test_compare_mvdr_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.set_variable('lookDirection', self.W_pca)
        mlab.run_code("W = bss.beamformer.mvdr('noiseMatrix', Phi_NN, 'lookDirection', lookDirection);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_mvdr)

    @tc.attr.matlab
    def test_compare_gev_beamformer(self):
        mlab = self.mlab.process
        mlab.set_variable('Phi_XX', self.Phi_XX)
        mlab.set_variable('Phi_NN', self.Phi_NN)
        mlab.run_code("W = bss.beamformer.gev('cleanObservationMatrix', Phi_XX, 'noiseMatrix', Phi_NN);")
        W_matlab = mlab.get_variable('W')
        tc.assert_cosine_similarity(W_matlab, self.W_gev)

    @tc.attr.matlab
    def test_compare_lcmv_beamformer(self):
        from paderbox.speech_enhancement.beamformer import apply_beamforming_vector
        from paderbox.transform.module_stft import istft
        from paderbox.evaluation import sxr_module

        data = self.data_multi_speaker
        X = data['X']  # K F D T
        Y = data['Y']  # F D T
        N = data['N']  # F D T

        # Caculate masks
        # X_mask.shape = (2, 513, 128)
        # N_mask.shape = (513, 128)
        *X_mask, N_mask = wiener_like_mask([*X, N],
                                           source_axis=0,
                                           sensor_axis=-2)
        # Phi_XX.shape = (2, 513, 6, 6)
        # Phi_NN.shape = (513, 6, 6)
        Phi_XX = get_power_spectral_density_matrix(Y, X_mask, source_dim=0)  # K F D D
        Phi_NN = get_power_spectral_density_matrix(Y, N_mask)  # F D D

        # W_pca.shape = (2, 513, 6)
        W_pca = get_pca_vector(Phi_XX)

        # W_lcmv1.shape = (513, 6)
        # W_lcmv2.shape = (513, 6)
        # W_lcmv.shape = (2, 513, 6)
        W_lcmv1 = get_lcmv_vector(W_pca, [1, 0], Phi_NN)
        W_lcmv2 = get_lcmv_vector(W_pca, [0, 1], Phi_NN)
        W_lcmv = np.array(([W_lcmv1, W_lcmv2]))

        # W_pca_tmp.shape = (513, 2, 6)
        W_pca_tmp = W_pca.transpose(1, 2, 0)
        Phi_NN_tmp = Phi_NN

        mlab = self.mlab
        mlab.set_variable('W_pca', W_pca_tmp)
        mlab.set_variable('Phi_NN', Phi_NN_tmp)
        # mlab.run_code_print('size(W_pca)')
        # mlab.run_code_print('size(Phi_NN)')
        mlab.run_code(
            "[W(:, :, 1), failing] = bss.beamformer.lcmv('observationMatrix', Phi_NN, 'lookDirection', W_pca, 'responseVector', [1 0]);")
        mlab.run_code(
            "W(:, :, 2) = bss.beamformer.lcmv('observationMatrix', Phi_NN, 'lookDirection', W_pca, 'responseVector', [0 1]);")
        W_matlab = mlab.get_variable('W')
        failing = mlab.get_variable('failing')
        print(np.sum(failing))
        assert (np.sum(failing) == 0.0)

        def sxr_output(W):
            num_frames = 128
            try:
                Shat = np.zeros((2, 2, num_frames, 513), dtype=complex)
                Shat[0, 0, :, :] = apply_beamforming_vector(W[0, :, :], X[0, :, :, :]).T
            except ValueError:
                num_frames = 316
                Shat = np.zeros((2, 2, 513, num_frames), dtype=complex)
                Shat[0, 0, :, :] = apply_beamforming_vector(W[0, :, :], X[0, :, :, :]).T
            Shat[1, 1, :, :] = apply_beamforming_vector(W[1, :, :], X[1, :, :, :]).T
            Shat[0, 1, :, :] = apply_beamforming_vector(W[1, :, :], X[0, :, :, :]).T
            Shat[1, 0, :, :] = apply_beamforming_vector(W[0, :, :], X[1, :, :, :]).T

            Nhat = np.zeros((2, num_frames, 513), dtype=complex)
            Nhat[0, :, :] = apply_beamforming_vector(W[0, :, :], N).T
            Nhat[0, :, :] = apply_beamforming_vector(W[0, :, :], N).T
            shat = istft(Shat)
            nhat = istft(Nhat)
            return sxr_module.output_sxr(shat, nhat)

        W_matlab_tmp = W_matlab.transpose(2, 0, 1)
        W_lcmv_tmp = W_lcmv

        sxr_matlab = sxr_output(W_matlab_tmp)
        sxr_py = sxr_output(W_lcmv_tmp)

        tc.assert_almost_equal(sxr_matlab, sxr_py)

        tc.assert_cosine_similarity(W_matlab_tmp, W_lcmv_tmp)


if __name__ == '__main__':
    unittest.main()