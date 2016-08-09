import unittest

import numpy
import scipy
import scipy.signal

import nt.io.audioread as io
import nt.reverb.reverb_utils as rirUtils
import nt.reverb.scenario as scenario
import nt.testing as tc
from nt.io.data_dir import testing as testing_dir
from nt.utils.matlab import Mlab, matlab_test

# Uncomment, if you want to test Matlab functions.
matlab_test = unittest.skipUnless(True, 'matlab-test')


def time_convolve(x, impulse_response):
    """
    Takes audio signals and the impulse responses according to their position
    and returns the convolution. The number of audio signals in x are required
    to correspond to the number of sources in the given RIR.
    Convolution is conducted through time domain.

    :param x: [number_sources x audio_signal_length - array] the audio signal
        to convolve
    :param impulse_response:
        [filter_length x number_sensors x number_sources - numpy matrix ]
        The three dimensional impulse response.
    :return: convolved_signal:
        [number_sources x number_sensors x signal_length - numpy matrix]
        The convoluted signal for every sensor and each source
    """
    _, sensors, sources = impulse_response.shape

    if not sources == x.shape[0]:
        raise Exception(
            "Number audio signals (" +
            str(x.shape[0]) +
            ") does not match source positions (" +
            str(sources) +
            ") in given impulse response!"
        )
    convolved_signal = numpy.zeros(
        [sources, sensors, x.shape[1] + len(impulse_response) - 1]
    )

    for i in range(sensors):
        for j in range(sources):
            convolved_signal[j, i, :] = numpy.convolve(
                x[j, :],
                impulse_response[:, i, j]
            )

    return convolved_signal



class TestReverbUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.matlab_session = Mlab()
        self.sample_rate = 16000  # Hz
        self.filter_length = 2 ** 13
        self.room_dimensions = (10, 10, 4)  # meter
        self.sensor_pos = (5.01, 5, 2)
        self.soundvelocity = 343

    # @unittest.skip("")
    @matlab_test
    def test_comparePythonTranVuRirWithExpectedUsingMatlabTwoSensorTwoSrc(self):
        """
        Compare RIR calculated by Matlabs reverb.generate(..) "Tranvu"
        algorithm with RIR calculated by Python reverb_utils.generate_RIR(..)
        "Tranvu" algorithm.
        Here: 2 randomly placed sensors and sources each
        """
        number_of_sources = 2
        number_of_sensors = 2
        reverberation_time = 0.1

        sources, mics = scenario.generate_uniformly_random_sources_and_sensors(
            self.room_dimensions,
            number_of_sources,
            number_of_sensors
        )

        matlab_session = self.matlab_session
        pyRIR = rirUtils.generate_RIR(
            self.room_dimensions,
            sources,
            mics,
            self.sample_rate,
            self.filter_length,
            reverberation_time
        )

        matlab_session.run_code("roomDim = [{0}; {1}; {2}];".format(
            self.room_dimensions[0],
            self.room_dimensions[1],
            self.room_dimensions[2])
        )
        matlab_session.run_code("src = zeros(3,1); sensors = zeros(3,1);")
        for s in range(number_of_sources):
            matlab_session.run_code("srctemp = [{0};{1};{2}];".format(
                sources[s][0],
                sources[s][1],
                sources[s][2])
            )
            matlab_session.run_code("src = [src srctemp];")
        for m in range(number_of_sensors):
            matlab_session.run_code("sensorstemp = [{0};{1};{2}];".format(
                mics[m][0],
                mics[m][1],
                mics[m][2])
            )
            matlab_session.run_code("sensors = [sensors sensorstemp];")

        matlab_session.run_code("src = src(:, 2:end);")
        matlab_session.run_code("sensors = sensors(:, 2:end);")

        matlab_session.run_code("sampleRate = {0};".format(self.sample_rate))
        matlab_session.run_code("filterLength = {0};".format(self.filter_length))
        matlab_session.run_code("T60 = {0};".format(reverberation_time))

        matlab_session.run_code(
            "rir = reverb.generate(roomDim, src, sensors, sampleRate, " +
            "filterLength, T60, 'algorithm', 'TranVu');"
        )

        matlabRIR = matlab_session.get_variable('rir')
        tc.assert_allclose(matlabRIR, pyRIR, atol=1e-4)

    def test_compareTranVuMinimumTimeDelayWithSoundVelocity(self):
        """
        Compare theoretical TimeDelay from distance and soundvelocity with
        timedelay found via index of maximum value in calculated RIR.
        Here: 1 Source, 1 Sensor, no reflections, that is, T60 = 0
        """
        numSrcs = 1
        numMics = 1
        T60 = 0

        sources, mics = scenario.generate_uniformly_random_sources_and_sensors(
            self.room_dimensions,
            numSrcs,
            numMics
        )
        distance = numpy.linalg.norm(
            numpy.asarray(sources) - numpy.asarray(mics))

        # Tranvu: first index of returned RIR equals time-index minus 128
        fixedshift = 128
        RIR = rirUtils.generate_RIR(
            self.room_dimensions,
            sources,
            mics,
            self.sample_rate,
            self.filter_length,
            T60
        )
        peak = numpy.argmax(RIR) - fixedshift
        actual = peak / self.sample_rate
        expected = distance / 343
        tc.assert_allclose(actual, expected, atol=1e-4)

    @matlab_test
    def test_compare_tran_vu_expected_T60_with_schroeder_method(self):
        """
        Compare minimal time-delay of RIR calculated by TranVu's algorithm
        with expected propagation-time by given distance and soundvelocity.

        Similarity ranges between 0.1 and 0.2 difference depending on given
        T60.
        """
        number_of_sources = 1
        number_of_sensors = 1
        T60 = 0.2

        sources, mics = scenario.generate_uniformly_random_sources_and_sensors(
            self.room_dimensions,
            number_of_sources,
            number_of_sensors
        )
        # By using TranVu the first index of returned RIR equals time-index -128
        fixedshift = 128

        rir = rirUtils.generate_RIR(self.room_dimensions,
                                    sources,
                                    mics,
                                    self.sample_rate,
                                    self.filter_length,
                                    T60)

        if number_of_sources == 1:
            rir = numpy.reshape(rir, (self.filter_length, 1))
            assert rir.shape == (self.filter_length, 1)

        matlab_session = self.matlab_session
        matlab_session.run_code("sampleRate = {0};".format(self.sample_rate))
        matlab_session.run_code("fixedShift = {0};".format(fixedshift))
        matlab_session.run_code("rir = zeros({0},{1},{2});".format(
            self.filter_length, number_of_sensors, number_of_sources))
        codeblock = ""
        for m in rir:
            codeblock += "{0};".format(m)
        codeblock = codeblock[:-1]  # omit last comma
        matlab_session.run_code("rir = [" + codeblock + "];")
        matlabRIR = matlab_session.get_variable('rir')
        matlab_session.run_code(
            "actual = RT_schroeder(rir(fixedShift+1:end)',sampleRate);"
        )
        actualT60 = matlab_session.get_variable('actual')

        tc.assert_allclose(matlabRIR, rir, atol=1e-4)
        tc.assert_allclose(actualT60, T60, atol=0.14)

    def test_TranVu_cardioid_sensororient_zero(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from cardioid directivity and zero sensor orientation with
        expected characteristic.
        """
        sensor_orientation = 0
        algorithm = "TranVu"
        sensor_directivity = "cardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    def test_TranVu_subcardioid_sensororient_zero(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from subcardioid directivity and zero sensor orientation
        with expected characteristic.
        """
        sensor_orientation = 0
        algorithm = "TranVu"
        sensor_directivity = "subcardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    def test_TranVu_cardioid_sensororient_piover4(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from cardioid directivity and pi over 4 sensor orientation
        with expected characteristic.
        """
        sensor_orientation = numpy.pi / 4
        algorithm = "TranVu"
        sensor_directivity = "cardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    def test_TranVu_subcardioid_sensororient_piover4(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from subcardioid directivity and pi over 4 sensor orientation
        with expected characteristic.
        """
        sensor_orientation = numpy.pi / 4
        algorithm = "TranVu"
        sensor_directivity = "subcardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    def test_TranVu_cardioid_sensororient_piover2(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from cardioid directivity and pi over 2 sensor orientation
        with expected characteristic.
        """
        sensor_orientation = numpy.pi / 2
        algorithm = "TranVu"
        sensor_directivity = "cardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    def test_TranVu_subcardioid_sensororient_piover2(self):
        """
        Compare signal-power's max and min of RIR calculated by TranVu's
        algorithm from subcardioid directivity and pi over 2 sensor orientation
        with expected characteristic.
        """
        sensor_orientation = numpy.pi / 2
        algorithm = "TranVu"
        sensor_directivity = "subcardioid"

        self.process_with_directivity_characteristic(
            algorithm=algorithm,
            sensor_orientation_angle=sensor_orientation,
            sensor_directivity=sensor_directivity
        )

    # @unittest.skip("")
    @matlab_test
    def test_compare_mlab_conv_pyOverlap_Save(self):
        """
        based on original from
        http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python

        :return:
        """
        f0 = 440  # 440 Hz
        fs = 8000  # sampled at 8 kHz
        T = 2  # lasting 5 seconds

        # Create test signal and STFT.
        t = numpy.linspace(0, T, T * fs, endpoint=False)
        x = numpy.sin(2 * scipy.pi * f0 * t)
        b = numpy.random.rand(250)
        y_hat = numpy.convolve(x, b, "full")
        # y_overlap_save = rirUtils.conv_overlap_save(x,b)
        y_convolved = scipy.signal.fftconvolve(x, b, "full")

        matlab_session = self.matlab_session
        codeblock = ""
        for m in b:
            codeblock += "{0};".format(m)
        codeblock = codeblock[:-1]  # omit last comma
        matlab_session.run_code("bshrt = [" + codeblock + "];")

        codeblock = ""
        for m in x:
            codeblock += "{0};".format(m)
        codeblock = codeblock[:-1]  # omit last comma
        matlab_session.run_code("x = [" + codeblock + "];")
        matlab_session.run_code("y = conv(bshrt,x);")
        matlab_y = matlab_session.get_variable("y")

        # test compare conv in time domain with conv_overlapSave
        tc.assert_allclose(y_convolved, y_hat, atol=1e-4)

    def process_with_directivity_characteristic(
            self, algorithm="TranVu",
            sensor_orientation_angle=0,
            sensor_directivity="omnidirectional"
    ):
        actual_maxmin, expected_maxmin = self.get_directivity_characteristic(
            algorithm=algorithm,
            angle="azimuth",
            sensor_orientation_angle=sensor_orientation_angle,
            directivity=sensor_directivity
        )
        tc.assert_allclose(actual_maxmin, expected_maxmin, atol=1e-5)

        actual_maxmin, expected_maxmin = self.get_directivity_characteristic(
            algorithm=algorithm,
            angle="elevation",
            sensor_orientation_angle=sensor_orientation_angle,
            directivity=sensor_directivity
        )
        tc.assert_allclose(actual_maxmin, expected_maxmin, atol=1e-5)

    def get_directivity_characteristic(self, algorithm="TranVu",
                                       angle="azimuth",
                                       sensor_orientation_angle=0,
                                       directivity="cardioid"):
        if algorithm == "TranVu":
            fixedShift = 128
            T60 = 0
        elif algorithm == "Habets":
            fixedShift = 0
            T60 = 0.18
        else:
            fixedShift = 0
            T60 = 0
        deltaAngle = numpy.pi / 16

        if angle == "azimuth":
            azimuth_angle = numpy.arange(0, 2 * numpy.pi, deltaAngle)
            elevation_angle = numpy.zeros([int(2 * numpy.pi / deltaAngle)])
            sensor_orientations = [[sensor_orientation_angle, 0], ]
        elif angle == "elevation":
            azimuth_angle = numpy.zeros([int(2 * numpy.pi / deltaAngle)])
            elevation_angle = numpy.arange(0, 2 * numpy.pi, deltaAngle)
            sensor_orientations = [[0, sensor_orientation_angle], ]
        else:
            raise NotImplementedError("Given angle not implemented!"
                                      "Choose 'azimuth' or 'elevation'!")
        radius = 1
        sources_position = \
            scenario.generate_deterministic_source_positions(
                center=self.sensor_pos,
                n=len(azimuth_angle),
                azimuth_angles=azimuth_angle,
                elevation_angles=elevation_angle,
                radius=radius,
                dims=3
            )

        rir_py = rirUtils.generate_RIR(self.room_dimensions,
                                       sources_position,
                                       [self.sensor_pos, ],
                                       self.sample_rate,
                                       self.filter_length,
                                       T60,
                                       algorithm="TranVu",
                                       sensorDirectivity=directivity,
                                       sensorOrientations=sensor_orientations
                                       )
        # set RIR to 0 as of the point in time where we're receiving echoes
        image_distance = numpy.array(self.room_dimensions * 6).reshape((3, 6))
        filter = numpy.array([[1, 0, 0, -1, 0, 0],
                              [0, 1, 0, 0, -1, 0],
                              [0, 0, 1, 0, 0, -1]]
                             )
        image_distance *= filter
        first_source_images = sources_position.transpose().reshape(
            [3, 1, -1, 1]) + \
                              image_distance.reshape([3, 1, 1, -1])
        distance_sensor_images = numpy.sqrt(((numpy.asarray(
            self.sensor_pos).reshape([3, 1, 1, 1]) - first_source_images) ** 2)
                                            .sum(axis=0))

        minimum_distance_sensor_image = numpy.min(distance_sensor_images)
        minimum_echo_time_delay = \
            minimum_distance_sensor_image / self.soundvelocity
        minimum_echo_time_index = numpy.floor(
            minimum_echo_time_delay * self.sample_rate + fixedShift)
        cutoff_index = minimum_echo_time_index

        rir_py[cutoff_index - 1:, :, :] = numpy.zeros([
            self.filter_length - cutoff_index + 1,
            rir_py.shape[1],
            int(rir_py.shape[2])])

        # calculate squared power for each source
        squared_power = numpy.sum(rir_py * numpy.conjugate(rir_py), axis=0)
        squared_power = numpy.reshape(squared_power, [len(azimuth_angle), 1])

        max_index = numpy.argmax(squared_power)
        min_index = numpy.argmin(squared_power)
        actual = numpy.array([(max_index) * deltaAngle,
                              (min_index) * deltaAngle])
        if directivity == "cardioid" or directivity == "subcardioid":
            expected = numpy.array([sensor_orientation_angle,
                                    sensor_orientation_angle + numpy.pi])
        else:
            raise NotImplementedError(
                "Test is not runnable for directivities " +
                "other than cardioid or subcardioid. " +
                "Selected directivity: " + directivity
            )
        return actual, expected

    def test_convolution(self):
        # Check whether convolution through frequency domain via fft yields the
        # same as through time domain.
        testsignal1 = io.audioread(
            testing_dir('timit', 'data', 'sample_1.wav'))
        testsignal2 = io.audioread(
            testing_dir('timit', 'data', 'sample_1.wav'))
        testsignal3 = io.audioread(
            testing_dir('timit', 'data', 'sample_1.wav'))
        # pad all audiosignals with zeros such they have equal lengths
        maxlen = numpy.amax((len(testsignal1),
                             len(testsignal2),
                             len(testsignal3)
                             ))
        testsignal1 = numpy.pad(testsignal1,
                                (0, maxlen - len(testsignal1)),
                                'constant')
        testsignal2 = numpy.pad(testsignal2,
                                (0, maxlen - len(testsignal2)),
                                'constant')
        testsignal3 = numpy.pad(testsignal3,
                                (0, maxlen - len(testsignal3)),
                                'constant')
        audio = numpy.vstack([testsignal1,
                              testsignal2,
                              testsignal3]
                             )
        [sources_positions, mic_positions] = scenario. \
            generate_uniformly_random_sources_and_sensors(
            self.room_dimensions,
            3,
            8
        )
        T60 = 0.3
        rir_py = rirUtils.generate_RIR(
            self.room_dimensions,
            sources_positions,
            mic_positions,
            self.sample_rate,
            self.filter_length,
            T60,
            algorithm="TranVu"
        )
        convolved_signal_fft = rirUtils.convolve(audio, rir_py.T)
        convolved_signal_time = time_convolve(audio, rir_py)
        tc.assert_allclose(
            convolved_signal_fft,
            convolved_signal_time,
            atol=1e-10
        )
