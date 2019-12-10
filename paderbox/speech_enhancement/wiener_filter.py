import numpy as np


def _spectrogram(x):
    # return np.sum(x.astype(np.complex128).
    # view(np.float64).resahpe(*x.shape, 2)**2, axis=-1)

    return x.real ** 2 + x.imag ** 2


def wiener_filter_gain_apriori(observation, s_mask, n_mask, G_min_db=-25,
                               mask_min=1e-6):
    """
        Decision directed approach Wiener filter implementation
         as used in [1].

        [1] Chinaev, Heitkaemper, Häb-Umbach, "A Priori SNR Estimation
         Using Weibull Mixture Model", 12. ITG Fachtagung
        Sprachkommunikation (ITG 2016). 2016

        :param observation: Single channel complex STFT signal with
            dimensions TxF.
        :type observation: ndarray
        :param s_mask: SPP like mask.
        :type s_mask: ndarray
        :param n_mask: NPP like mask.
        :type n_mask: ndarray
        :param G_min_db: Minimal Gain in dB. Defaults to -25.
        :type G_min_db: int, optional
        :param mask_min: Minimal mask value. Defaults to 1e-6.
        :type mask_min: float, optional

        :return: Estimate for the Wiener filter gain TxF using a apriori
        SNR estimation similar to the decision directed approach.
        :rtype: ndarray


        >>> obs = np.array([1, 1, 1, 1, 1])[:, None]
        >>> mask = np.array([1, 1, 1, 1, 1])[:, None]
        >>> G_min_db = -25
        >>> (wiener_filter_gain(obs, mask) - 10 ** (G_min_db / 20)).ravel()
        array([0., 0., 0., 0., 0.])
        >>> mask = np.array([1, 0, 0, 0, 0])[:, None]
        >>> (wiener_filter_gain(obs, mask) - 10 ** (G_min_db / 20)).ravel()
        array([0., 0., 0., 0., 0.])
        """
    G_min = 10 ** (G_min_db / 20)
    n_mask = np.clip(n_mask, mask_min, (1 - mask_min))
    s_mask = np.clip(s_mask, mask_min, (1 - mask_min))
    Phi_SS = _spectrogram(s_mask * observation)
    Phi_NN = _spectrogram(n_mask * observation)

    Phi_NN = np.maximum(Phi_NN, mask_min)
    a_priori = Phi_SS / Phi_NN
    a_priori_SNR = np.maximum(a_priori, mask_min)
    gain = np.maximum(1 / (1 + 1 / a_priori_SNR), G_min)
    return gain


def wiener_filter_gain(observation, n_mask, G_min_db=-25, mask_min=1e-6):
    """
        Wiener filter implementation as suggested in the DSSP script.
        This function returns the gain function, which is multiplied with the
        STFT of the noisy observation.

        The NPP-based noise PSD estimation is proposed in [1].

        [1] Chinaev, Heymann, Drude, Häb-Umbach, "Noise-Presence-Probability-
        Based Noise PSD Estimation by Using DNNs", 12. ITG Fachtagung
        Sprachkommunikation (ITG 2016). 2016

        :param observation: Single channel complex STFT signal with
            dimensions TxF.
        :type observation: ndarray
        :param n_mask: NPP like mask.
        :type n_mask: ndarray
        :param G_min_db: Minimal Gain in dB. Defaults to -25.
        :type G_min_db: int, optional
        :param mask_min: Minimal mask value. Defaults to 1e-6.
        :type mask_min: float, optional

        :return: Estimate for the Wiener filter gain TxF.
        :rtype: ndarray


        >>> obs = np.array([1, 1, 1, 1, 1])[:, None]
        >>> mask = np.array([1, 1, 1, 1, 1])[:, None]
        >>> G_min_db = -25
        >>> (wiener_filter_gain(obs, mask) - 10 ** (G_min_db / 20)).ravel()
        array([0., 0., 0., 0., 0.])
        >>> mask = np.array([1, 0, 0, 0, 0])[:, None]
        >>> (wiener_filter_gain(obs, mask) - 10 ** (G_min_db / 20)).ravel()
        array([0., 0., 0., 0., 0.])
        """

    G_min = 10 ** (G_min_db / 20)
    n_mask = np.clip(n_mask, mask_min, (1 - mask_min))
    Phi_XX = _spectrogram(observation)
    Phi_NN = _spectrogram(n_mask * observation)
    Phi_NN_smoothed = Phi_NN
    for t in range(1, n_mask.shape[0]):
        Phi_NN_smoothed[t, :] = (1 - n_mask[t, :]) * \
            Phi_NN_smoothed[t - 1, :] + Phi_NN[t, :]

    Phi_NN_smoothed = np.maximum(Phi_NN_smoothed, mask_min)
    a_posteriori_SNR = Phi_XX / Phi_NN_smoothed
    a_posteriori_SNR = np.maximum(a_posteriori_SNR, mask_min)
    gain = np.maximum((1 - 1 / a_posteriori_SNR), G_min)
    return gain


def wiener_filter(observation, n_mask, G_min_db=-25, mask_min=1e-6):
    """
    Apply gain function calculated via the a posteriori SNR as suggested
    in the DSSP script.

    :param observation: Single channel complex STFT signal with
            dimensions TxF.
    :type observation: ndarray
    :param n_mask: NPP like mask.
    :type n_mask: ndarray
    :param G_min_db: Minimal Gain in dB. Defaults to -25.
    :type G_min_db: int, optional
    :param mask_min: Minimal mask value. Defaults to 1e-6.
    :type mask_min: float, optional

    :return: Estimate for the speech signal STFT with dimensions TxF.
    :rtype: ndarray
    """

    gain = wiener_filter_gain(observation, n_mask, G_min_db=G_min_db,
                              mask_min=mask_min)
    return gain * observation

if __name__ == '__main__':
    import asn
    from paderbox.speech_enhancement.mask_module import wiener_like_mask
    from paderbox.transform.module_stft import stft_v2 as stft, istft_v2 as istft
    from paderbox.io import play, audiowrite
    from paderbox.speech_enhancement.noise import get_snr
    from paderbox.evaluation import pesq

    dataset = asn.database.chime.Dataset_dt05_simu()
    data = dataset[0]
    obs = stft(data.observed[4])
    x = stft(data.X[4])
    n = stft(data.N[4])
    _, n_mask = wiener_like_mask([x, n])
    s_hat = wiener_filter(obs, n_mask)
    g = wiener_filter_gain(obs, n_mask)
    # play.play(s_hat)
    audiowrite.audiowrite(
        istft(s_hat),
        '/net/nas/boeddeker/asn_data/wiener_test.wav')
    print('snr', get_snr(istft(x * g), istft(n * g)))
    print('pesq', pesq.pesq(data.X[4], istft(s_hat)))
