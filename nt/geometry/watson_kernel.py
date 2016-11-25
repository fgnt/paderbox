import numpy
from nt.speech_enhancement.beamform_utils import get_nearfield_time_of_flight
from nt.speech_enhancement.beamform_utils import get_stft_center_frequencies
from nt.speech_enhancement.beamform_utils import get_steering_vector


def watson_kernel_doa_estimation(
        Y, true_src_pos, source_positions, sensor_positions,
        filter_length, sample_rate, stft_size, room_Dim, T60
):
    """
    Generates the Complex Watson Kernel averaged PDF

    :param room_Dim 3-floats-sequence; The room dimensions in meters
    :param true_src_pos with shape [1, 3]
    :param sensor_positions List of 3-floats (#sensors). The sensor
        positions in meter within room dimensions.
    :param sample_rate as constant
    :param filter_length number of filter coefficients
    :param T60 sound decay time
    :return j averaged PDF as List of complex floats
    """
    tau = get_nearfield_time_of_flight(source_positions, sensor_positions.T)
    f = get_stft_center_frequencies(stft_size, sample_rate)

    # Anechoic unit-norm normalized model
    W=get_steering_vector(tau,stft_size, sample_rate, True)

    # Watson PDF
    _, frames, frequencies = Y.shape
    kappa = 1
    p = numpy.einsum('abd,bcd->acd', W.conj(), Y)
    p = kappa * (p.real**2 + p.imag**2)
    p = numpy.exp(p)
    p = numpy.sum(p, axis=(1, 2))

    return p / frequencies / frames
