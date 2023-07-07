import numpy as np
from paderbox.transform.module_stft import STFT


def _griffin_lim_step(
    x: np.ndarray,
    reconstruction_stft: np.ndarray,
    stft: STFT
):
    reconstruction_angle = np.angle(reconstruction_stft)
    # Discard magnitude part of the reconstruction and use the supplied
    # magnitude spectrogram instead.
    proposal_spec = x * np.exp(1.0j * reconstruction_angle)
    audio = stft.inverse(proposal_spec)
    reconstruction_stft = stft(audio)

    return reconstruction_stft, audio


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
    reconstruction_stft = stft(audio)
    for n in range(iterations):
        reconstruction_stft, audio = _griffin_lim_step(
            x, reconstruction_stft, stft
        )

        if verbose:
            reconstruction_magnitude = np.abs(reconstruction_stft)
            diff = (
                np.linalg.norm(x - reconstruction_magnitude, ord='fro')
                / (np.linalg.norm(x, ord='fro') + 1e-5)
            )  # Spectral Convergence
            print(
                'Reconstruction iteration: {}/{} SC: {} dB'.format(
                    n, iterations, 10 * np.log10(diff)
                )
            )
    return audio


def fast_griffin_lim(
    x: np.ndarray,
    stft: STFT,
    alpha=0.99,
    iterations=100,
    verbose=False,
):
    """Griffin-Lim algorithm with momentum for phase retrieval [1, 2].

    Usually has a faster convergence than the original Griffin-Lim algorithm
    and may converge to a better local optimum.

    [1]: Perraudin, Nathanaël, Peter Balazs, and Peter L. Søndergaard. "A fast
        Griffin-Lim algorithm." 2013 IEEE Workshop on Applications of Signal
        Processing to Audio and Acoustics. IEEE, 2013.
    [2]: Peer, Tal, Simon Welker, and Timo Gerkmann. "Beyond Griffin-LIM:
        Improved Iterative Phase Retrieval for Speech." 2022 International
        Workshop on Acoustic Signal Enhancement (IWAENC). IEEE, 2022.

    Args:
        x: Magnitude spectrogram of shape (*, num_frames, stft.size//2+1)
        stft: paderbox.transform.module_stft.STFT instance
        alpha: Momentum for GLA acceleration, where 0 <= alpha <= 1
        iterations: Number of optimization iterations
        verbose: If True, print the reconstruction error after each iteration
            step

    >>> f_0 = 200
    >>> f_s = 16_000
    >>> t = np.linspace(0, 1, num=f_s)
    >>> sine = np.sin(2*np.pi*f_0*t)
    >>> sine.shape
    (16000,)
    >>> stft = STFT(200, 1024, window_length=800, fading=False, pad=True)
    >>> x = stft(sine)
    >>> x.shape
    (77, 513)
    >>> reconstruction = fast_griffin_lim(np.abs(x), stft, iterations=5)
    >>> reconstruction.shape
    (16000,)
    """

    if not 0. <= alpha <= 1.:
        raise ValueError(f'alpha must be in [0, 1], but is {alpha}.')

    # Random phase initialization
    angle = np.random.uniform(low=-np.pi, high=np.pi, size=x.shape)
    reconstruction_stft = x * np.exp(1.0j * angle)

    y = reconstruction_stft  # Stores accelerated STFT reconstruction
    for n in range(iterations):
        rec_stft_, audio = _griffin_lim_step(x, y, stft)
        y = rec_stft_ + alpha * (rec_stft_ - reconstruction_stft)  # Momentum
        reconstruction_stft = rec_stft_
        if verbose:
            reconstruction_magnitude = np.abs(reconstruction_stft)
            diff = (
                np.linalg.norm(x - reconstruction_magnitude, ord='fro')
                / (np.linalg.norm(x, ord='fro') + 1e-5)
            )  # Spectral Convergence
            print(
                'Reconstruction iteration: {}/{} SC: {} dB'.format(
                    n, iterations, 10 * np.log10(diff)
                )
            )

    return audio
