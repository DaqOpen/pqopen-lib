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

class TestPowerPowerQualityUnbalance(unittest.TestCase):
    def setUp(self):
        ...
    
    def test_no_unbalance(self):
        u1 = 100*np.exp(1j*0*np.pi/180)
        u2 = 100*np.exp(1j*(-120)*np.pi/180)
        u3 = 100*np.exp(1j*120*np.pi/180)
        u0, u2 = pq.calc_unbalance([u1, u2, u3])

        self.assertAlmostEqual(u0, 0)
        self.assertAlmostEqual(u2, 0)

    def test_reverse_sequence(self):
        u1 = 100*np.exp(1j*0*np.pi/180)
        u2 = 100*np.exp(1j*(120)*np.pi/180)
        u3 = 100*np.exp(1j*(-120)*np.pi/180)
        u0, u2 = pq.calc_unbalance([u1, u2, u3])

        self.assertGreater(u2, 100)

    def test_moderate_unbalance(self):
        u1 = 100*np.exp(1j*0*np.pi/180)
        u2 = 90*np.exp(1j*(-120)*np.pi/180)
        u3 = 100*np.exp(1j*(120)*np.pi/180)
        u0, u2 = pq.calc_unbalance([u1, u2, u3])

        self.assertAlmostEqual(u0, 3.4482758620689)
        self.assertAlmostEqual(u2, 3.4482758620689)

    def test_zero_sequence(self):
        u1 = 100*np.exp(1j*0*np.pi/180) + 10
        u2 = 100*np.exp(1j*(-120)*np.pi/180) + 10
        u3 = 100*np.exp(1j*(120)*np.pi/180) + 10
        u0, u2 = pq.calc_unbalance([u1, u2, u3])

        self.assertAlmostEqual(u0, 10)
        self.assertAlmostEqual(u2, 0)

class TestPowerPowerQualityMsv(unittest.TestCase):
    def setUp(self):
        ...
    
    def test_no_msv(self):
        samplerate = 10000
        f_fund = 50
        duration = 0.2
        t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)

        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*f_fund*t)

        u_fft_rms = np.fft.rfft(u_values)/len(u_values)*np.sqrt(2)

        u_msv_rms = pq.calc_mains_signaling_voltage(u_fft_rms, 100, 10, 50)

        self.assertAlmostEqual(u_msv_rms, 0)

    def test_moderate_msv_50Hz(self):
        samplerate = 10000
        f_fund = 50
        duration = 0.2
        t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)

        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*f_fund*t) + np.sqrt(2)*np.sin(2*np.pi*275*t)

        u_fft_rms = np.fft.rfft(u_values)/len(u_values)*np.sqrt(2)

        u_msv_rms = pq.calc_mains_signaling_voltage(u_fft_rms, 275, 10, 50)

        self.assertAlmostEqual(u_msv_rms, 1)

    def test_moderate_msv_51Hz(self):
        samplerate = 10000
        f_fund = 51
        duration = 10.0/f_fund
        t = np.linspace(0, duration, int(samplerate*duration), endpoint=False)

        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*f_fund*t) + np.sqrt(2)*np.sin(2*np.pi*275*t)

        u_fft_rms = np.fft.rfft(u_values)/len(u_values)*np.sqrt(2)

        u_msv_rms = pq.calc_mains_signaling_voltage(u_fft_rms, 275, 10, f_fund)

        self.assertAlmostEqual(u_msv_rms, 1, places=2)

class TestPowerPowerQualityUnderOverDev(unittest.TestCase):
    def setUp(self):
        ...
    
    def test_under_deviation(self):
        u_rms = np.ones(10)
        u_rms[:5] *= 0.7875
        u_under = pq.calc_under_deviation(u_rms, 1)

        self.assertAlmostEqual(u_under, 10, places=2)

    def test_over_deviation(self):
        u_rms = np.ones(10)
        u_rms[:5] *= 1.1917
        u_over = pq.calc_over_deviation(u_rms, 1)

        self.assertAlmostEqual(u_over, 10, places=2)

class TestPowerPowerQualityMsvTracer(unittest.TestCase):
    def setUp(self):
        self.samplerate = 5000
        self.f_fund = 50.0
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, self.samplerate, endpoint=False)
        self.u1_base = np.sqrt(2)*np.sin(2*np.pi*self.f_fund*self.t)
    
    def test_simple_pattern(self):
        msv_tracer = pq.MainsSignalingVoltageTracer(
            samplerate=self.samplerate,
            bp_lo_cutoff_freq=373,
            bp_hi_cutoff_freq=393,
            lp_cutoff_freq=20,
            trigger_level=0.01,
            filter_order=4)
        u1 = self.u1_base
        u1[1000:2000] += 0.02*np.sqrt(2)*np.sin(2*np.pi*383*self.t[1000:2000])
        expected_msv_bit_list = [(14, 1), (24, 0)]
        samples_per_period = int(self.samplerate/self.f_fund)
        msv_bit_list = []
        for period_idx in range(int(self.f_fund/self.duration)):
            msv_bit, _ = msv_tracer.process(u1[period_idx*samples_per_period:(period_idx+1)*samples_per_period])
            if msv_bit is not None:
                msv_bit_list.append((period_idx, msv_bit))
        
        self.assertEqual(expected_msv_bit_list, msv_bit_list)

         
if __name__ == "__main__":
    unittest.main()