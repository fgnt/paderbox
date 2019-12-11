import unittest

from pb_bss.evaluation.module_stoi import *

from paderbox.io.audioread import audioread
import paderbox.testing as tc
from paderbox.testing.testfile_fetcher import get_file_path
from paderbox.utils.matlab import Mlab


# ToDo: move this code to pb_bss

@unittest.skip('To be moved to pb_bss')
class TestSTOI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = get_file_path("sample.wav")
        cls.x = audioread(path)[0]
        cls.sample_rate = 16000

    def test_stoi_range(self):
        processed = (self.x + 0.5*np.random.rand(1, self.x.shape[0])).flatten()
        d = stoi(self.x, processed, self.sample_rate)
        tc.assert_(0 <= d <= 1, "STOI value has to be in range (0, 1)")

    @tc.attr.matlab
    def test_taa_corr_matlab(self):
        x = np.random.rand(1, 20)
        y = np.random.rand(1, 20)

        # python
        python_corr = taa_corr(x, y)

        # matlab
        mlab = Mlab().process
        mlab.set_variable("x", x)
        mlab.set_variable("y", y)
        mlab.run_code("""
            xn    	= x-mean(x);
            xn  	= xn/sqrt(sum(xn.^2));
            yn   	= y-mean(y);
            yn    	= yn/sqrt(sum(yn.^2));
            rho   	= sum(xn.*yn);
        """)
        mlab_corr = mlab.get_variable("rho")

        # test
        tc.assert_almost_equal(python_corr, mlab_corr)

    @tc.attr.matlab
    def test_thirdoct_matlab(self):
        sample_rate = 10000
        nfft = 1024
        number_of_bands = 15
        first_center_frequency = 150
        mlab = Mlab().process

        # Matlab code
        mlab.set_variable("fs", sample_rate)
        mlab.set_variable("N_fft", nfft)
        mlab.set_variable("numBands", number_of_bands)
        mlab.set_variable("mn", first_center_frequency)
        mlab.run_code("""
            f = linspace(0, fs, N_fft+1);
            f = f(1:(N_fft/2+1));
            k = 0:(numBands-1);
            cf = 2.^(k/3)*mn;
            fl = sqrt((2.^(k/3)*mn).*2.^((k-1)/3)*mn);
            fr = sqrt((2.^(k/3)*mn).*2.^((k+1)/3)*mn);
            A = zeros(numBands, length(f));

            for i = 1:(length(cf))
                [a b] = min((f-fl(i)).^2);
                fl(i) = f(b);
                fl_ii = b;

                [a b] = min((f-fr(i)).^2);
                fr(i) = f(b);
                fr_ii = b;
                A(i,fl_ii:(fr_ii-1))= 1;
            end

            rnk         = sum(A, 2);
            numBands  	= find((rnk(2:end)>=rnk(1:(end-1))) & (rnk(2:end)~=0)~=0, 1, 'last' )+1;
            A           = A(1:numBands, :);
            cf          = cf(1:numBands);
            """)

        mlab_A = mlab.get_variable("A")
        mlab_cf = mlab.get_variable("cf")

        # python
        (python_A, python_cf) = thirdoct(sample_rate, nfft, number_of_bands,
                                         first_center_frequency)

        # test
        tc.assert_equal(mlab_cf.flatten().shape, python_cf.shape)
        tc.assert_equal(mlab_A.shape, python_A.shape)
        tc.assert_almost_equal(mlab_cf.flatten(), python_cf)
        tc.assert_almost_equal(mlab_A, python_A)

    @tc.attr.matlab
    def test_stoi(self):
        processed = (self.x + 0.5*np.random.rand(1, self.x.shape[0])).flatten()
        mlab = Mlab().process

        # Matlab
        mlab.set_variable("clean_signal", self.x)
        mlab.set_variable("processed_signal", processed)
        mlab.set_variable("sample_rate", self.sample_rate)
        mlab.run_code(
            "d = se.stoi(clean_signal, processed_signal, sample_rate);")
        mlab_d = mlab.get_variable("d")

        # Python
        python_d = stoi(self.x, processed, self.sample_rate)

        # test
        tc.assert_almost_equal(python_d, mlab_d, 1)
