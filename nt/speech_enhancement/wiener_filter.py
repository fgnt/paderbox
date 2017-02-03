import numpy as np


def wiener_filter_gain(observation, n_mask, G_min_db=-25, mask_min=1e-6):
    """
        Wiener filter implementation as suggested in the DSSP script.
        This function returns the gain function, which is multiplied with the
        STFT of the noisy obseration.

        The NPP-based noise PSD estimation is proposed in [1].

        Chinaev, Heymann, Drude, Häb-Umbach, "Noise-Presence-Probability-Based
        Noise PSD Estimation by Using DNNs", 12. ITG Fachtagung
        Sprachkommunikation (ITG 2016). 2016

        :param observation: Single channel complex STFT signal with
            dimensions TxF.
        :param n_mask:
        :param G_min_db: Minimal Gain in dB. Defaults to -25.
        :type G_min_db: int, optional
        :return: Estimate for the Wiener filter gain TxF.
        """

    def spectrogram(x):
        # return np.sum(x.astype(np.complex128).
        # view(np.float64).resahpe(*x.shape, 2)**2, axis=-1)

        return x.real ** 2 + x.imag ** 2

    G_min = 10 ** (G_min_db / 20)
    n_mask = np.clip(n_mask, mask_min, (1 - mask_min))
    Phi_XX = spectrogram(observation)
    Phi_NN = spectrogram(n_mask * observation)
    Phi_NN_smoothed = Phi_NN
    Phi_NN_smoothed[1:, :] = (1 - n_mask[1:, :]) * Phi_NN[:-1, :] \
                             + Phi_NN[1:, :]
    a_posteriori_SNR = Phi_XX / Phi_NN_smoothed
    gain = np.maximum((1 - 1 / a_posteriori_SNR), G_min)
    return gain


def wiener_filter(observation, n_mask, G_min_db=-25, mask_min=1e-6):
    """
    Apply gain function calculated via the a posteriori SNR as suggested
    in the DSSP script.

    :param observation: Single channel complex STFT signal with dimensions TxF.
    :param n_mask:
    :param G_min_db: Minimal Gain in dB. Defaults to -25.
    :type G_min_db: int, optional
    :return: Estimate for the speech signal STFT with dimensions TxF.
    """

    gain = wiener_filter_gain(observation, n_mask, G_min_db=G_min_db,
                              mask_min=mask_min)
    return gain * observation
