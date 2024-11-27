import unittest
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pqopen.powerquality as pq

class TestPowerPowerQualityHarmonic(unittest.TestCase):
    def setUp(self):
        ...

    def test_simple(self):
        samplerate = 1000
        f_fund = 50.0
        num_periods = 10
        t = np.linspace(0, 0.2, samplerate, endpoint=False)
        values = np.sqrt(2)*np.sin(2*np.pi*f_fund*t) + 0.1*np.sqrt(2)*np.sin(2*np.pi*2*f_fund*t + 45*np.pi/2)
        expected_v_h_rms = [0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]

        v_fft = pq.resample_and_fft(values)
        v_h_rms, v_h_phi = pq.calc_harmonics(v_fft, 10, 10)

        self.assertIsNone(np.testing.assert_allclose(v_h_rms, expected_v_h_rms, atol=0.01))
        self.assertAlmostEqual(v_h_phi[1], -90, places=2)
        self.assertAlmostEqual(v_h_phi[2], -90+45*2, places=2)

    def test_advanced(self):
        samplerate = 1000
        f_fund = 53.0
        num_periods = 10
        t = np.linspace(0, num_periods/f_fund, samplerate, endpoint=False)
        values = (1.0*np.sqrt(2)*np.sin(2*np.pi*1.0*f_fund*t) + 
                  0.1*np.sqrt(2)*np.sin(2*np.pi*1.5*f_fund*t + 45*np.pi/2)+
                  0.1*np.sqrt(2)*np.sin(2*np.pi*2.0*f_fund*t + 45*np.pi/2))
        expected_v_h_rms = [0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_v_ih_rms = [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_v_thd = 10.0


        v_fft = pq.resample_and_fft(values)
        v_h_rms, v_h_phi = pq.calc_harmonics(v_fft, 10, 10)
        v_ih_rms = pq.calc_interharmonics(v_fft, 10, 10)
        v_thd = pq.calc_thd(v_h_rms)


        self.assertIsNone(np.testing.assert_allclose(v_h_rms, expected_v_h_rms, atol=0.01))
        self.assertIsNone(np.testing.assert_allclose(v_ih_rms, expected_v_ih_rms, atol=0.01))
        self.assertAlmostEqual(v_h_phi[1], -90, places=2)
        self.assertAlmostEqual(v_h_phi[2], -90+45*2, places=2)
        self.assertAlmostEqual(v_thd, expected_v_thd, places=1)

         
if __name__ == "__main__":
    unittest.main()