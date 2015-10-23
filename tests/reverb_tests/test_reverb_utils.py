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
        self.sensorPositions = ((5.01, 5, 2),)
        self.sourcePositions = ((4,4,3.9),(8,8,0.1))


    @matlab_test
    def test_comparePythonTranVuRirWithExpectedUsingMatlabOneSensorOneSrc(self):


        T60 = 0
        sources,mics = rirUtils.generateRandomSourcesAndSensors(self.roomDim,
                                                                1,1)
        mlab = self.mlab
        pyRIR = rirUtils.generate_RIR(self.roomDim,sources,mics,self.sampleRate,
                                      self.filterLength,T60)
        mlab.set_variable('roomDim',"[{0};{1};{2}]".format(self.roomDim[0],
                                           self.roomDim[1],self.roomDim[2]))
        mlab.set_variable('src',"[{0};{1};{2}]".format(sources[0][0],
                                                      sources[0][1],
                                                      sources[0][2]))
        mlab.set_variable("sensors","[{0};{1};{2}]".format(mics[0][0],
                                                           mics[0][1],
                                                           mics[0][2]))
        mlab.set_variable("sampleRate","{0}".format(self.sampleRate))
        mlab.set_variable("filterLength","{0}".format(self.filterLength))
        mlab.set_variable("T60","0")

        mlab.run_code("rir = reverb.generate(roomDim,src,sensors,sampleRate,"+
                     "filterLength,T60,'algorithm','TranVu');")
        matlabRIR = mlab.get_variable('rir')

        tc.assert_allclose(matlabRIR,pyRIR, atol = 1e-4)
        #print(matlabRIR)
        #print(pyRIR)
        tc.assert_almost_equal(matlabRIR, pyRIR)



    @matlab_test
    def test_compareTranVuMinimumTimeDelayWithSoundVelocity(self):
        pass

    @matlab_test
    def test_compareTranVuExpectedT60WithCalculatedUsingSchroederMethod(self):
        pass

    @matlab_test
    def test_compareDirectivityWithExpectedUsingTranVu(self):
        pass

    @matlab_test
    def test_compareAzimuthSensorOrientationWithExpectedUsingTranVu(self):
        pass

    @matlab_test
    def test_compareElevationSensorOrientationWithExpectedUsingTranvu(self):
        pass

    @matlab_test
    def test_compareTranVuExpectedT60WithCalculatedUsingSchroederMethod(self):
        pass