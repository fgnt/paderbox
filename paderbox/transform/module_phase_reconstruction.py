import numpy as np
from paderbox.transform.module_stft import STFT


def griffin_lim(x, stft: STFT, iterations=100, verbose=False):
    """
    Reconstructs phase from magnitudes using Griffin-Lim algorithm and returns
    audio signal in time domain.

    Args:
        x: STFT Magnitudes (..., T, F)
        stft:
        iterations:
        verbose:

    Returns: audio signal

    >>> stft = STFT(160, 512, fading=False, pad=True)
    >>> audio_data=np.zeros(512 + 49*160)
    >>> x = stft(audio_data)
    >>> x.shape
    (50, 257)
    >>> reconstruction = griffin_lim(np.abs(x), stft, iterations=5)
    >>> reconstruction.shape
    (8352,)
    """
    nframes = x.shape[-2]
    nsamples = int(stft.frames_to_samples(nframes))
    # Initialize the reconstructed signal.
    audio = np.random.randn(nsamples)
    for n in range(iterations):
        reconstruction_stft = stft(audio)
        reconstruction_angle = np.angle(reconstruction_stft)
        # Discard magnitude part of the reconstruction and use the supplied
        # magnitude spectrogram instead.
        proposal_spec = x * np.exp(1.0j * reconstruction_angle)
        audio = stft.inverse(proposal_spec)

        if verbose:
            reconstruction_magnitude = np.abs(reconstruction_stft)
            diff = (np.sqrt(np.mean((reconstruction_magnitude - x) ** 2)))
            print(
                'Reconstruction iteration: {}/{} RMSE: {} '.format(
                    n, iterations, diff
                )
            )
    return audio
