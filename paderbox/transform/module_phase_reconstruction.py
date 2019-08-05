"""
This file contains the STFT function and related helper functions.
"""
import numpy as np
from paderbox.transform.module_stft import STFT


class GriffinLim:
    def __init__(self, stft: STFT):
        """
        Reconstructs phase from magnitudes using Griffin-Lim algorithm.

        Args:
            stft:

        >>> stft = STFT(160, 512, fading=False, pad=True)
        >>> audio_data=np.zeros(512 + 49*160)
        >>> x = stft(audio_data)
        >>> x.shape
        (50, 257)
        >>> griffin_lim = GriffinLim(stft)
        >>> reconstruction = griffin_lim(np.abs(x), iterations=5)
        >>> reconstruction.shape
        (8352,)
        """
        self.stft = stft

    def __call__(self, x, iterations=100, verbose=False):
        """

        Args:
            x: STFT Magnitudes (..., T, F)
            iterations:
            verbose:

        Returns:

        """
        nframes = x.shape[-2]
        nsamples = int(self.stft.frames_to_samples(nframes))
        # Initialize the reconstructed signal.
        audio = np.random.randn(nsamples)
        for n in range(iterations):
            reconstruction_stft = self.stft(audio)
            reconstruction_angle = np.angle(reconstruction_stft)
            # Discard magnitude part of the reconstruction and use the supplied
            # magnitude spectrogram instead.
            proposal_spec = x * np.exp(1.0j * reconstruction_angle)
            audio = self.stft.inverse(proposal_spec)

            if verbose:
                reconstruction_magnitude = np.abs(reconstruction_stft)
                diff = (np.sqrt(np.mean((reconstruction_magnitude - x) ** 2)))
                print(
                    'Reconstruction iteration: {}/{} RMSE: {} '.format(
                        n, iterations, diff
                    )
                )
        return audio
