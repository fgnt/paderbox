import unittest

import nt.io.audioread as io
import nt.reverb.reverb_utils as reverb_utils
import nt.testing as tc
import numpy as np
import nt.reverb.scenario as scenario
import scipy
import scipy.signal
from nose_parameterized import parameterized
from nt.io.data_dir import testing as testing_dir
from nt.utils.matlab import Mlab


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
    convolved_signal = np.zeros(
        [sources, sensors, x.shape[1] + len(impulse_response) - 1]
    )

    for i in range(sensors):
        for j in range(sources):
            convolved_signal[j, i, :] = np.convolve(
                x[j, :],
                impulse_response[:, i, j]
            )

    return convolved_signal


# TODO: Investigate CalcRIR_Simple_C.pyx and check lines 151 following.
#       Directional microphones seem to be broken.


class TestRoomImpulseGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = np.asarray([[10], [10], [4]])  # m
        cls.source_positions = np.asarray([[1, 1.1], [1, 1.1], [1.5, 1.5]])
        cls.sensor_positions = np.asarray([[2.2, 2.3], [2.4, 2.5], [1.4, 1.5]])
        cls.sample_rate = 16000  # Hz
        cls.filter_length = 2 ** 10
        cls.sound_decay_time = 0.5
        cls.sound_velocity = 343

    def test_compare_tran_vu_python_with_tran_vu_cython(self):
        rir_python = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=self.sound_decay_time,
            sound_velocity=self.sound_velocity,
            algorithm='tran_vu_python'
        )
        rir_cython = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=self.sound_decay_time,
            sound_velocity=self.sound_velocity,
            algorithm='tran_vu_cython'
        )
        np.testing.assert_allclose(
            rir_python, rir_cython, atol=1e-9
        )

    def test_compare_tran_vu_python_loopy_with_tran_vu_cython(self):
        rir_python = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=self.sound_decay_time,
            sound_velocity=self.sound_velocity,
            algorithm='tran_vu_python_loopy'
        )
        rir_cython = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=self.sound_decay_time,
            sound_velocity=self.sound_velocity,
            algorithm='tran_vu_cython'
        )
        np.testing.assert_allclose(
            rir_python, rir_cython, atol=1e-9
        )

    @parameterized.expand(reverb_utils.available_rir_algorithms)
    def test_compare_time_delay_with_sound_velocity(self, algorithm):
        """
                Compare theoretical TimeDelay from distance and soundvelocity
                with timedelay found via index of maximum value in calculated
                RIR.
                Here: 1 Source, 1 Sensor, no reflections, that is, T60 = 0
                """
        T60 = 0

        # one source and one sensor
        source_positions = self.source_positions[:, 0:1]
        sensor_positions = self.sensor_positions[:, 0:1]

        distance = np.linalg.norm(
            source_positions - sensor_positions)

        # Tranvu: first index of returned RIR equals time-index minus 128
        fixedshift = 128
        rir = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=source_positions,
            sensor_positions=sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=T60,
            sound_velocity=self.sound_velocity,
            algorithm=algorithm
        )
        peak = np.argmax(rir) - fixedshift
        actual = peak / self.sample_rate
        expected = distance / self.sound_velocity
        tc.assert_allclose(actual, expected, atol=1e-4)

    def _test_compare_rir_with_matlab(self,
                                      reverberation_time=0.1,
                                      algorithm='tran_vu_python'):
        """
        Compare RIR calculated by Matlabs reverb.generate(..) "Tranvu"
        algorithm with RIR calculated by Python reverb_utils.generate_RIR(..)
        "Tranvu" algorithm.
        Here: 2 randomly placed sensors and sources each
        """
        number_of_sources = self.source_positions.shape[1]
        number_of_sensors = self.sensor_positions.shape[1]

        matlab_session = Mlab().process
        py_rir = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=self.source_positions,
            sensor_positions=self.sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=reverberation_time,
            algorithm=algorithm
        )

        matlab_session.run_code("roomDim = [{0[0]}; {0[1]}; {0[2]}];".format(
            self.room)
        )
        matlab_session.run_code("src = zeros(3,1); sensors = zeros(3,1);")
        for s in range(number_of_sources):
            matlab_session.run_code("srctemp = [{0};{1};{2}];".format(
                self.source_positions[0][s],
                self.source_positions[1][s],
                self.source_positions[2][s])
            )
            matlab_session.run_code("src = [src srctemp];")
        for m in range(number_of_sensors):
            matlab_session.run_code("sensorstemp = [{0};{1};{2}];".format(
                self.sensor_positions[0][m],
                self.sensor_positions[1][m],
                self.sensor_positions[2][m])
            )
            matlab_session.run_code("sensors = [sensors sensorstemp];")

        matlab_session.run_code("src = src(:, 2:end);")
        matlab_session.run_code("sensors = sensors(:, 2:end);")

        matlab_session.run_code("sampleRate = {0};".format(self.sample_rate))
        matlab_session.run_code(
            "filterLength = {0};".format(self.filter_length))
        matlab_session.run_code("T60 = {0};".format(reverberation_time))

        matlab_session.run_code(
            "rir = reverb.generate(roomDim, src, sensors, sampleRate, " +
            "filterLength, T60, 'algorithm', 'TranVu');"
        )

        mlab_rir = matlab_session.get_variable('rir')
        tc.assert_allclose(mlab_rir, py_rir.T, atol=1e-4)

    @parameterized.expand(reverb_utils.available_rir_algorithms,
                          lambda f, num, p: '{}_{}'.format(f.__name__,
                                                           str(p.args[0])))
    @tc.attr.matlab
    def test_compare_matlab_with(self, algorithm):
        print(algorithm)
        self._test_compare_rir_with_matlab(algorithm=algorithm)

    @parameterized.expand(reverb_utils.available_rir_algorithms)
    @tc.attr.matlab
    def test_compare_expected_T60_with_schroeder_method(self, algorithm):
        """
        Compare minimal time-delay of RIR calculated by TranVu's algorithm
        with expected propagation-time by given distance and soundvelocity.

        Similarity ranges between 0.1 and 0.2 difference depending on given
        T60.
        """

        # get positions for one source and one sensor
        source_positions = self.source_positions[:, 0:1]
        sensor_positions = self.sensor_positions[:, 0:1]
        number_of_sources = source_positions.shape[1]
        number_of_sensors = sensor_positions.shape[1]
        T60 = 0.2

        # By using TranVu the first index of returned RIR equals time-index -128
        fixedshift = 128

        rir = reverb_utils.generate_rir(room_dimensions=self.room,
                                        source_positions=source_positions,
                                        sensor_positions=sensor_positions,
                                        sample_rate=self.sample_rate,
                                        filter_length=self.filter_length,
                                        sound_decay_time=T60,
                                        algorithm=algorithm)

        if number_of_sources == 1:
            rir = np.reshape(rir, (self.filter_length, 1))
            assert rir.shape == (self.filter_length, 1)

        matlab_session = Mlab().process
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


class TestConvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.room = np.asarray([[10], [10], [4]])  # m
        cls.source_positions = np.asarray([[1, 1.1], [1, 1.1], [1.5, 1.5]])
        cls.sensor_positions = np.asarray([[2.2, 2.3], [2.4, 2.5], [1.4, 1.5]])
        cls.sample_rate = 16000  # Hz
        cls.filter_length = 2 ** 10
        cls.sound_decay_time = 0.5
        cls.sound_velocity = 343

    @tc.attr.matlab
    def test_compare_mlab_conv_pyOverlap_save(self):
        """
        based on original from
        http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python

        :return:
        """
        f0 = 440  # 440 Hz
        fs = 8000  # sampled at 8 kHz
        T = 2  # lasting 5 seconds

        # Create test signal and STFT.
        t = np.linspace(0, T, T * fs, endpoint=False)
        x = np.sin(2 * scipy.pi * f0 * t)
        b = np.random.rand(250)
        y_hat = np.convolve(x, b, "full")
        # y_overlap_save = rirUtils.conv_overlap_save(x,b)
        y_convolved = scipy.signal.fftconvolve(x, b, "full")

        matlab_session = Mlab().process
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

    @parameterized.expand(reverb_utils.available_rir_algorithms)
    def test_convolution(self, algorithm):
        # Check whether convolution through frequency domain via fft yields the
        # same as through time domain.

        # get audio data
        testsignal1 = io.audioread(
            testing_dir / 'timit' / 'data' / 'sample_1.wav')
        testsignal2 = io.audioread(
            testing_dir / 'timit' / 'data' / 'sample_1.wav')
        testsignal3 = io.audioread(
            testing_dir / 'timit' / 'data' / 'sample_1.wav')
        # pad all audiosignals with zeros such they have equal lengths
        maxlen = np.amax((len(testsignal1),
                          len(testsignal2),
                          len(testsignal3)
                          ))
        testsignal1 = np.pad(testsignal1,
                             (0, maxlen - len(testsignal1)),
                             'constant')
        testsignal2 = np.pad(testsignal2,
                             (0, maxlen - len(testsignal2)),
                             'constant')
        testsignal3 = np.pad(testsignal3,
                             (0, maxlen - len(testsignal3)),
                             'constant')
        audio = np.vstack([testsignal1,
                           testsignal2,
                           testsignal3]
                          )

        # get source positions for three audio sources and mic positions for
        # 8 mics
        [source_positions, sensor_positions] = scenario. \
            generate_uniformly_random_sources_and_sensors(
            self.room,
            3,
            8
        )
        source_positions = np.asarray(source_positions).T[0]
        sensor_positions = np.asarray(sensor_positions).T[0]

        T60 = 0.3
        rir_py = reverb_utils.generate_rir(
            room_dimensions=self.room,
            source_positions=source_positions,
            sensor_positions=sensor_positions,
            sample_rate=self.sample_rate,
            filter_length=self.filter_length,
            sound_decay_time=T60,
            algorithm=algorithm
        )
        convolved_signal_fft = reverb_utils.convolve(audio, rir_py)
        convolved_signal_time = time_convolve(audio, rir_py.T)
        tc.assert_allclose(
            convolved_signal_fft,
            convolved_signal_time,
            atol=1e-10
        )
