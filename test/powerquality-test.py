import unittest
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pqopen.powerquality as pq
from daqopen.channelbuffer import DataChannelBuffer

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


class TestPowerPowerQualityFlicker(unittest.TestCase):
    def setUp(self):
        ...

    def test_steady_state(self):
        samplerate = 10000
        f_fund = 50
        duration = 120
        t = np.linspace(0, duration, samplerate*duration, endpoint=False)
        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*f_fund*t)

        voltage_fluctuation = pq.VoltageFluctuation(samplerate)

        blocksize = 1000
        for blk_idx in range(t.size // blocksize):
            hp_data = 230*np.ones(samplerate//blocksize*f_fund*2)
            voltage_fluctuation.process(blk_idx*blocksize, hp_data, u_values[blk_idx*blocksize:(blk_idx+1)*blocksize])
        pst = voltage_fluctuation.calc_pst(0, duration*samplerate)

        self.assertGreaterEqual(voltage_fluctuation._pinst_channel.sample_count, 1000*40)
        self.assertAlmostEqual(pst, 0.0, places=1)

    def test_230V_50Hz_8Hz8(self):
        samplerate = 10000
        f_fund = 50
        duration = 30
        t = np.linspace(0, duration, samplerate*duration, endpoint=False)
        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*f_fund*t)
        f_mod = 8.8
        d_mod = 0.00250/2
        u_values *= (1+d_mod*np.sin(2*np.pi*f_mod*t))

        voltage_fluctuation = pq.VoltageFluctuation(samplerate)

        blocksize = 1000
        for blk_idx in range(t.size // blocksize):
            hp_data = 230*np.ones(samplerate//blocksize*f_fund*2)
            voltage_fluctuation.process(blk_idx*blocksize, hp_data, u_values[blk_idx*blocksize:(blk_idx+1)*blocksize])
        pinst_1s, _ = voltage_fluctuation._pinst_channel.read_data_by_acq_sidx((duration-1)*samplerate, duration*samplerate)

        self.assertGreaterEqual(voltage_fluctuation._pinst_channel.sample_count, 1000*(duration-20))
        self.assertAlmostEqual(pinst_1s.max(), 1.0, places=2)


         
if __name__ == "__main__":
    unittest.main()