import numpy
import unittest
from nt.utils.matlab import Mlab, matlab_test
import nt.testing as tc
import numpy.testing as nptest
import nt.reverb.reverb_utils as rirUtils

# uncomment, if you want to test matlab functions
matlab_test = unittest.skipUnless(True,'matlab-test')

class TestReverbUtils(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.mlab = Mlab()
        self.sampleRate = 16000 # Hz
        self.filterLength = 2**13
        self.roomDim = (10,10,4) # meter

    @matlab_test
    def test_comparePythonTranVuRirWithExpectedUsingMatlabTwoSensorTwoSrc(self):
        """
        Compare RIR calculated by Matlabs reverb.generate(..) "Tranvu"
        algorithm with RIR calculated by Python reverb_utils.generate_RIR(..)
        "Tranvu" algorithm.
        Here: 2 randomly placed sensors and sources each
        """
        numSrcs = 2
        numMics = 2
        T60 = 0.1

        sources,mics = rirUtils.generateRandomSourcesAndSensors(self.roomDim,
                                                                numSrcs,numMics)
        mlab = self.mlab
        pyRIR = rirUtils.generate_RIR(self.roomDim,sources,mics,self.sampleRate,
                                      self.filterLength,T60)

        mlab.run_code("roomDim = [{0};{1};{2}]".format(self.roomDim[0],
                                          self.roomDim[1],self.roomDim[2]))
        mlab.run_code("src = zeros(3,1); sensors = zeros(3,1);")
        for s in range(numSrcs):
            mlab.run_code("srctemp = [{0};{1};{2}]".format(sources[s][0],
                                                      sources[s][1],
                                                      sources[s][2]))
            mlab.run_code("src = [src srctemp]")
        for m in range(numMics):
            mlab.run_code("sensorstemp = [{0};{1};{2}]".format(mics[m][0],
                                                           mics[m][1],
                                                           mics[m][2]))
            mlab.run_code("sensors = [sensors sensorstemp]")
        mlab.run_code("src = src(:,2:end)")
        mlab.run_code("sensors = sensors(:,2:end)")

        mlab.run_code("sampleRate = {0}".format(self.sampleRate))
        mlab.run_code("filterLength = {0}".format(self.filterLength))
        mlab.run_code("T60 = {0}".format(T60))

        mlab.run_code("rir = reverb.generate(roomDim,src,sensors,sampleRate,"+
                     "filterLength,T60,'algorithm','TranVu');")
        matlabRIR = mlab.get_variable('rir')
        tc.assert_allclose(matlabRIR,pyRIR, atol = 1e-4)

    def test_compareTranVuMinimumTimeDelayWithSoundVelocity(self):
        """
        Compare theoretical TimeDelay from distance and soundvelocity with
        timedelay found via index of maximum value in calculated RIR.
        Here: 1 Source, 1 Sensor, no reflections, that is, T60 = 0
        """
        numSrcs = 1
        numMics = 1
        T60 = 0

        sources,mics = rirUtils.generateRandomSourcesAndSensors(self.roomDim,
                                                                numSrcs,numMics)
        distance = numpy.linalg.norm(numpy.asarray(sources)-numpy.asarray(mics))

        fixedshift = 128 #Tranvu: first index of returned RIR equals time-index
                        # minus 128
        RIR = rirUtils.generate_RIR(self.roomDim,sources,mics,self.sampleRate,
                                    self.filterLength,T60)
        peak = numpy.argmax(RIR) - fixedshift
        actual = peak / self.sampleRate;
        expected = distance / 343
        tc.assert_allclose(actual,expected, atol = 1e-4)

    @unittest.skip("")
    @matlab_test
    def test_compareTranVuExpectedT60WithCalculatedUsingSchroederMethod(self):
        pass
    @unittest.skip("")
    @matlab_test
    def test_compareDirectivityWithExpectedUsingTranVu(self):
        pass
    @unittest.skip("")
    @matlab_test
    def test_compareAzimuthSensorOrientationWithExpectedUsingTranVu(self):
        pass
    @unittest.skip("")
    @matlab_test
    def test_compareElevationSensorOrientationWithExpectedUsingTranvu(self):
        pass
    @unittest.skip("")
    @matlab_test
    def test_compareTranVuExpectedT60WithCalculatedUsingSchroederMethod(self):
        pass