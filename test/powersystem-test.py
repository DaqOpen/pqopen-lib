import unittest
import os
import sys
import numpy as np
import datetime
from unittest.mock import MagicMock
from pathlib import Path
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from daqopen.channelbuffer import AcqBuffer
from pqopen.powersystem import PowerSystem, PowerPhase

class TestPowerSystemChannelConfig(unittest.TestCase):
    def setUp(self):
        # Mock AcqBuffer
        self.mock_acq_buffer = MagicMock()
        self.mock_time_channel = MagicMock()

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.mock_acq_buffer,
            input_samplerate=1000.0
        )

    def test_initialization(self):
        self.assertEqual(self.power_system._samplerate, 1000.0)
        self.assertEqual(self.power_system.nominal_frequency, 50.0)
        self.assertEqual(len(self.power_system._phases), 0)

    def test_add_phase(self):
        self.power_system.add_phase(u_channel=self.mock_acq_buffer, name="Phase A")
        self.assertEqual(len(self.power_system._phases), 1)
        self.assertEqual(self.power_system._phases[0].name, "Phase A")

    def test_enable_harmonic_calculation(self):
        self.power_system.enable_harmonic_calculation(num_harmonics=10)
        self.assertEqual(self.power_system._features["harmonics"], 10)

    def test_enable_fluctuation_calculation(self):
        self.power_system.enable_fluctuation_calculation()
        self.assertTrue(self.power_system._features["fluctuation"])

    def test_update_calc_channels(self):
        self.power_system.add_phase(u_channel=self.mock_acq_buffer, name="u")
        self.power_system._update_calc_channels()
        self.assertTrue("Uu_1p_rms" in self.power_system.output_channels)

    def test_process_method(self):
        # Assuming process is to be implemented, we just check its existence
        self.assertTrue(callable(self.power_system.process))

class TestPowerPhaseChannelConfig(unittest.TestCase):
    def setUp(self):
        # Mock AcqBuffer
        self.mock_u_channel = MagicMock()
        self.mock_i_channel = MagicMock()

        # Create PowerPhase instance
        self.power_phase = PowerPhase(
            u_channel=self.mock_u_channel,
            i_channel=self.mock_i_channel,
            number=1,
            name="Phase A"
        )

    def test_initialization(self):
        self.assertEqual(self.power_phase.name, "Phase A")
        self.assertEqual(self.power_phase._number, 1)
        self.assertIsNotNone(self.power_phase._u_channel)
        self.assertIsNotNone(self.power_phase._i_channel)

    def test_update_calc_channels(self):
        features = {"harmonics": 10, "fluctuation": True}
        self.power_phase.update_calc_channels(features=features)
        self.assertIn("one_period", self.power_phase._calc_channels)
        self.assertIn("trms", self.power_phase._calc_channels["one_period"]["voltage"])
        self.assertIn("harm_rms", self.power_phase._calc_channels["multi_period"]["voltage"])


class TestPowerSystemZcd(unittest.TestCase):
    def setUp(self):
        self.u_channel = AcqBuffer()

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u_channel,
            input_samplerate=1000.0,
            zcd_threshold=0.1
        )
        # Add Phase
        self.power_system.add_phase(u_channel=self.u_channel)

    def test_zcd_normal(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        values = np.sin(2*np.pi*50*t)
        period = int(self.power_system._samplerate) // 50
        expected_zero_crossings = [period*cycle for cycle in range(50)]
        expected_frequency = np.array(expected_zero_crossings[2:])*0 + 50

        self.u_channel.put_data(values[:values.size//2])
        self.power_system.process()
        self.u_channel.put_data(values[values.size//2:])
        self.power_system.process()

        self.assertEqual(self.power_system._zero_cross_counter, 49)
        # Allow maximum deviation of 1 sample
        self.assertIsNone(np.testing.assert_array_almost_equal(self.power_system._zero_crossings, expected_zero_crossings[-20:], 0))

        # Check Frequency
        frequency, _ = self.power_system.output_channels["Freq"].read_data_by_acq_sidx(0, values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(frequency, expected_frequency, 2))

    def test_one_period_calc(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        values = np.sqrt(2)*np.sin(2*np.pi*50*t)
        
        expected_u_rms = np.array(np.zeros(48)) + 1.0

        self.u_channel.put_data(values[:values.size//2])
        self.power_system.process()
        self.u_channel.put_data(values[values.size//2:])
        self.power_system.process()

        # Check Voltage
        u_rms, _ = self.power_system.output_channels["U1_1p_rms"].read_data_by_acq_sidx(0, values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(u_rms, expected_u_rms, 3))

    def test_one_period_calc_fallback(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        values = t*0 + 1.0 # DC
        
        expected_u_rms = np.array(np.zeros(48)) + 1.0

        self.u_channel.put_data(values[:values.size//2])
        self.power_system.process()
        self.u_channel.put_data(values[values.size//2:])
        self.power_system.process()

        # Check Voltage
        u_rms, _ = self.power_system.output_channels["U1_1p_rms"].read_data_by_acq_sidx(0, values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(u_rms[-10:], expected_u_rms[-10:], 3))

class TestPowerSystemCalculation(unittest.TestCase):
    def setUp(self):
        self.u_channel = AcqBuffer()
        self.i_channel = AcqBuffer()

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u_channel,
            input_samplerate=10000.0,
            zcd_threshold=0.1
        )
        # Add Phase
        self.power_system.add_phase(u_channel=self.u_channel, i_channel=self.i_channel)

    def test_one_period_calc_single_phase(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u_values = np.sqrt(2)*np.sin(2*np.pi*50*t)
        i_values = 2*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5
        
        expected_u_rms = np.array(np.zeros(47)) + 1.0
        expected_i_rms = np.array(np.zeros(47)) + 2.0
        expected_p_avg = np.array(np.zeros(47)) + 1.0

        self.u_channel.put_data(u_values)
        self.i_channel.put_data(i_values)
        self.power_system.process()

        # Check Voltage
        u_rms, _ = self.power_system.output_channels["U1_1p_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(u_rms[1:], expected_u_rms, 3))
        # Check Current
        i_rms, _ = self.power_system.output_channels["I1_1p_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(i_rms[1:], expected_i_rms, 3))
        # Check Power
        p_avg, _ = self.power_system.output_channels["P1_1p"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p_avg[1:], expected_p_avg, 3))

    def test_multi_period_calc_single_phase(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u_values = np.sqrt(2)*np.sin(2*np.pi*50*t)
        i_values = 2*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5
        
        expected_u_rms = np.array(np.zeros(4)) + 1.0
        expected_i_rms = np.array(np.zeros(4)) + 2.0
        expected_p_avg = np.array(np.zeros(4)) + 1.0
        expected_sidx = np.arange(1,5) * 0.2 * self.power_system._samplerate + 0.02 * self.power_system._samplerate

        self.power_system.enable_harmonic_calculation(10)
        self.u_channel.put_data(u_values)
        self.i_channel.put_data(i_values)
        self.power_system.process()

        # Check Voltage
        u_rms, sidx = self.power_system.output_channels["U1_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_allclose(sidx, expected_sidx, atol=1))
        self.assertIsNone(np.testing.assert_allclose(u_rms, expected_u_rms, rtol=0.01))
        u_h_rms, _ = self.power_system.output_channels["U1_H_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_allclose(u_h_rms[:,1], expected_u_rms, rtol=0.01))
        # Check Current
        i_rms, sidx = self.power_system.output_channels["I1_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(i_rms, expected_i_rms, 3))
        i_h_rms, _ = self.power_system.output_channels["I1_H_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_allclose(i_h_rms[:,1], expected_i_rms, rtol=0.01))
        # Check Power
        p_avg, sidx = self.power_system.output_channels["P1"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p_avg, expected_p_avg, 3))

    def test_multi_period_calc_harmonic_msv(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u_values = np.sqrt(2)*np.sin(2*np.pi*50*t) + 0.1*np.sqrt(2)*np.sin(2*np.pi*150*t) + 0.01*np.sqrt(2)*np.sin(2*np.pi*375*t)
        i_values = 2*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5

        expected_u_h3_rms = np.array(np.zeros(4)) + 0.1
        expected_u_msv_rms = np.array(np.zeros(4)) + 0.01

        self.power_system.enable_harmonic_calculation(10)
        self.power_system.enable_mains_signaling_calculation(375)
        self.u_channel.put_data(u_values)
        self.i_channel.put_data(i_values)
        self.power_system.process()

        # Check Voltage        
        u_h_rms, _ = self.power_system.output_channels["U1_H_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_allclose(u_h_rms[:,3], expected_u_h3_rms, rtol=0.01))
        u_msv_rms, _ = self.power_system.output_channels["U1_msv_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertIsNone(np.testing.assert_allclose(u_msv_rms, expected_u_msv_rms, rtol=0.01))
        

class TestPowerSystemCalculationThreePhase(unittest.TestCase):
    def setUp(self):
        self.u1_channel = AcqBuffer()
        self.u2_channel = AcqBuffer()
        self.u3_channel = AcqBuffer()

        self.i1_channel = AcqBuffer()
        self.i2_channel = AcqBuffer()
        self.i3_channel = AcqBuffer()

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u1_channel,
            input_samplerate=10000.0,
            zcd_threshold=0.1
        )
        # Add Phases
        self.power_system.add_phase(u_channel=self.u1_channel, i_channel=self.i1_channel)
        self.power_system.add_phase(u_channel=self.u2_channel, i_channel=self.i2_channel)
        self.power_system.add_phase(u_channel=self.u3_channel, i_channel=self.i3_channel)

    def test_multi_period_calc(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        u2_values = 1.1*np.sqrt(2)*np.sin(2*np.pi*50*t - 120*np.pi/180)
        u3_values = 1.2*np.sqrt(2)*np.sin(2*np.pi*50*t + 120*np.pi/180)
        i1_values = 2.0*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5
        i2_values = 2.1*np.sqrt(2)*np.sin(2*np.pi*50*t+(-120+60)*np.pi/180) # cos_phi = 0.5
        i3_values = 2.2*np.sqrt(2)*np.sin(2*np.pi*50*t+(120+60)*np.pi/180) # cos_phi = 0.5
        
        expected_u1_rms = np.array(np.zeros(4)) + 1.0
        expected_u2_rms = np.array(np.zeros(4)) + 1.1
        expected_u3_rms = np.array(np.zeros(4)) + 1.2

        expected_i1_rms = np.array(np.zeros(4)) + 2.0
        expected_i2_rms = np.array(np.zeros(4)) + 2.1
        expected_i3_rms = np.array(np.zeros(4)) + 2.2

        expected_p1_avg = np.array(np.zeros(4)) + 1.0*2.0*0.5
        expected_p2_avg = np.array(np.zeros(4)) + 1.1*2.1*0.5
        expected_p3_avg = np.array(np.zeros(4)) + 1.2*2.2*0.5

        expected_p_avg = expected_p1_avg + expected_p2_avg + expected_p3_avg

        expected_q1_t = np.array(np.zeros(4)) + 1.0*2.0*np.sin(np.arccos(0.5))
        expected_q1_fund = np.array(np.zeros(4)) - 1.0*2.0*np.sin(np.arccos(0.5))

        expected_unbal_0 = np.array(np.zeros(4)) + 5.249
        expected_unbal_2 = np.array(np.zeros(4)) + 5.249

        expected_sidx = np.arange(1,5) * 0.2 * self.power_system._samplerate + 0.02 * self.power_system._samplerate

        self.power_system.enable_harmonic_calculation(10)
        self.u1_channel.put_data(u1_values)
        self.u2_channel.put_data(u2_values)
        self.u3_channel.put_data(u3_values)

        self.i1_channel.put_data(i1_values)
        self.i2_channel.put_data(i2_values)
        self.i3_channel.put_data(i3_values)

        self.power_system.process()

        # Check Voltage U1
        u1_rms, sidx = self.power_system.output_channels["U1_rms"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(u1_rms, expected_u1_rms, rtol=0.01))
        u1_h_rms, _ = self.power_system.output_channels["U1_H_rms"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(u1_h_rms[:,1], expected_u1_rms, rtol=0.01))
        # Check Voltage U2
        u2_rms, sidx = self.power_system.output_channels["U2_rms"].read_data_by_acq_sidx(0, u2_values.size)
        self.assertIsNone(np.testing.assert_allclose(u2_rms, expected_u2_rms, rtol=0.01))
        u2_h_rms, _ = self.power_system.output_channels["U2_H_rms"].read_data_by_acq_sidx(0, u2_values.size)
        self.assertIsNone(np.testing.assert_allclose(u2_h_rms[:,1], expected_u2_rms, rtol=0.01))
        # Check Voltage U3
        u3_rms, sidx = self.power_system.output_channels["U3_rms"].read_data_by_acq_sidx(0, u3_values.size)
        self.assertIsNone(np.testing.assert_allclose(u3_rms, expected_u3_rms, rtol=0.01))
        u3_h_rms, _ = self.power_system.output_channels["U3_H_rms"].read_data_by_acq_sidx(0, u3_values.size)
        self.assertIsNone(np.testing.assert_allclose(u3_h_rms[:,1], expected_u3_rms, rtol=0.01))
        # Check Current I1
        i1_rms, sidx = self.power_system.output_channels["I1_rms"].read_data_by_acq_sidx(0, i1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(i1_rms, expected_i1_rms, 3))
        i1_h_rms, _ = self.power_system.output_channels["I1_H_rms"].read_data_by_acq_sidx(0, i1_values.size)
        self.assertIsNone(np.testing.assert_allclose(i1_h_rms[:,1], expected_i1_rms, rtol=0.01))
        # Check Current I2
        i2_rms, sidx = self.power_system.output_channels["I2_rms"].read_data_by_acq_sidx(0, i2_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(i2_rms, expected_i2_rms, 3))
        i2_h_rms, _ = self.power_system.output_channels["I2_H_rms"].read_data_by_acq_sidx(0, i2_values.size)
        self.assertIsNone(np.testing.assert_allclose(i2_h_rms[:,1], expected_i2_rms, rtol=0.01))
        # Check Current I1
        i3_rms, sidx = self.power_system.output_channels["I3_rms"].read_data_by_acq_sidx(0, i3_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(i3_rms, expected_i3_rms, 3))
        i3_h_rms, _ = self.power_system.output_channels["I3_H_rms"].read_data_by_acq_sidx(0, i3_values.size)
        self.assertIsNone(np.testing.assert_allclose(i3_h_rms[:,1], expected_i3_rms, rtol=0.01))
        # Check Power P1
        p1_avg, sidx = self.power_system.output_channels["P1"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p1_avg, expected_p1_avg, 3))
        # Check Power P2
        p2_avg, sidx = self.power_system.output_channels["P2"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p2_avg, expected_p2_avg, 3))
        # Check Power P3
        p3_avg, sidx = self.power_system.output_channels["P3"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p3_avg, expected_p3_avg, 3))

        # Check Overall Power
        p_avg, sidx = self.power_system.output_channels["P"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(p_avg, expected_p_avg, 3))

        # Check Reactive Power
        # Check Q1_t
        q1_t, sidx = self.power_system.output_channels["Q1_t"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(q1_t, expected_q1_t, 3))
        # Check Q1_fund
        q1_fund, sidx = self.power_system.output_channels["Q1_H1"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(q1_fund, expected_q1_fund, 3))

        # Check Balance
        u0, sidx = self.power_system.output_channels["U_unbal_0"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(u0, expected_unbal_0, 3))
        u2, sidx = self.power_system.output_channels["U_unbal_2"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_array_almost_equal(u2, expected_unbal_2, 3))

    def test_energy_calc_pos(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        u2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t - 120*np.pi/180)
        u3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t + 120*np.pi/180)
        i1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        i2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t -120*np.pi/180) 
        i3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t+ 120*np.pi/180)

        Path(SCRIPT_DIR+"/data_files/energy.json").write_text(json.dumps({"W_pos": 10, "W_neg": 10}))

        expected_w_pos = np.array([1, 2, 3, 4])*3*0.2/3600 + 10
        expected_w_neg = np.array([1, 2, 3, 4])*0.0/3600 + 10

        self.power_system.enable_energy_channels(Path(SCRIPT_DIR+"/data_files/energy.json"))
        self.u1_channel.put_data(u1_values)
        self.u2_channel.put_data(u2_values)
        self.u3_channel.put_data(u3_values)

        self.i1_channel.put_data(i1_values)
        self.i2_channel.put_data(i2_values)
        self.i3_channel.put_data(i3_values)

        self.power_system.process()

        # Check Energy W_pos
        w_pos, sidx = self.power_system.output_channels["W_pos"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(w_pos, expected_w_pos, rtol=0.01))
        w_neg, sidx = self.power_system.output_channels["W_neg"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(w_neg, expected_w_neg, rtol=0.01))

    def test_energy_calc_neg(self):
        t = np.linspace(0, 1, int(self.power_system._samplerate), endpoint=False)
        u1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        u2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t - 120*np.pi/180)
        u3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t + 120*np.pi/180)
        i1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        i2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t -120*np.pi/180) 
        i3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t+ 120*np.pi/180)

        Path(SCRIPT_DIR+"/data_files/energy.json").write_text(json.dumps({"W_pos": 10, "W_neg": 10}))

        expected_w_pos = np.array([1, 2, 3, 4])*0.0/3600 + 10
        expected_w_neg = np.array([1, 2, 3, 4])*3*0.2/3600 + 10

        self.power_system.enable_energy_channels(Path(SCRIPT_DIR+"/data_files/energy.json"))
        self.u1_channel.put_data(u1_values)
        self.u2_channel.put_data(u2_values)
        self.u3_channel.put_data(u3_values)

        self.i1_channel.put_data(-i1_values)
        self.i2_channel.put_data(-i2_values)
        self.i3_channel.put_data(-i3_values)

        self.power_system.process()

        # Check Energy W_pos
        w_pos, sidx = self.power_system.output_channels["W_pos"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(w_pos, expected_w_pos, rtol=0.01))
        w_neg, sidx = self.power_system.output_channels["W_neg"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(w_neg, expected_w_neg, rtol=0.01))

    def test_power_calc_pos_neg(self):
        # 1 second positve power (add one period at the end to await sync)
        t = np.linspace(0, 1.02, int(self.power_system._samplerate*1.02), endpoint=False)
        u1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        u2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t - 120*np.pi/180)
        u3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t + 120*np.pi/180)
        i1_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)
        i2_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t -120*np.pi/180) 
        i3_values = 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t+ 120*np.pi/180)
        # 1 second negative power
        t += 1.02
        u1_values = np.r_[u1_values, 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)]
        u2_values = np.r_[u2_values, 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t - 120*np.pi/180)]
        u3_values = np.r_[u3_values, 1.0*np.sqrt(2)*np.sin(2*np.pi*50*t + 120*np.pi/180)]
        i1_values = np.r_[i1_values, -1.0*np.sqrt(2)*np.sin(2*np.pi*50*t)]
        i2_values = np.r_[i2_values, -1.0*np.sqrt(2)*np.sin(2*np.pi*50*t -120*np.pi/180)]
        i3_values = np.r_[i3_values, -1.0*np.sqrt(2)*np.sin(2*np.pi*50*t+ 120*np.pi/180)]

        expected_p_pos = np.array([3, 3, 3, 3, 3, 0, 0, 0, 0, 0])
        expected_p_neg = np.array([0, 0, 0, 0, 0, 3, 3, 3, 3, 3])

        self.u1_channel.put_data(u1_values)
        self.u2_channel.put_data(u2_values)
        self.u3_channel.put_data(u3_values)

        self.i1_channel.put_data(i1_values)
        self.i2_channel.put_data(i2_values)
        self.i3_channel.put_data(i3_values)

        self.power_system.process()

        # Check Power Pos
        p_pos, sidx = self.power_system.output_channels["P_pos"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(p_pos, expected_p_pos, rtol=0.01))
        p_neg, sidx = self.power_system.output_channels["P_neg"].read_data_by_acq_sidx(0, u1_values.size)
        self.assertIsNone(np.testing.assert_allclose(p_neg, expected_p_neg, rtol=0.01))

class TestPowerSystemNperSync(unittest.TestCase):
    def setUp(self):
        self.u_channel = AcqBuffer()
        self.i_channel = AcqBuffer()
        self.time_channel = AcqBuffer(dtype=np.int64)

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u_channel,
            input_samplerate=1000.0,
            zcd_threshold=0.1
        )
        # Add Phase
        self.power_system.add_phase(u_channel=self.u_channel, i_channel=self.i_channel)
        self.power_system.enable_nper_abs_time_sync(self.time_channel, interval_sec=10)

    def test_short_interval(self):
        abs_ts_start = datetime.datetime(2024,1,1,0,0,5, tzinfo=datetime.UTC).timestamp()
        t = np.linspace(0, 21, int(self.power_system._samplerate)*21, endpoint=False)
        u_values = np.sqrt(2)*np.sin(2*np.pi*50*t)
        i_values = 2*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5
        self.u_channel.put_data(u_values)
        self.i_channel.put_data(i_values)
        self.time_channel.put_data((t+abs_ts_start)*1e6)
        self.power_system.process()

        u_rms, sidx = self.power_system.output_channels["U1_rms"].read_data_by_acq_sidx(0, u_values.size)
        self.assertAlmostEqual(sidx[5*5],5.2*self.power_system._samplerate, places=-1)

    # def test_fractional_freq(self):
    #     abs_ts_start = datetime.datetime(2024,1,1,0,0,5, tzinfo=datetime.UTC).timestamp()
    #     t = np.linspace(0, 21, int(self.power_system._samplerate)*21, endpoint=False)
    #     u_values = np.sqrt(2)*np.sin(2*np.pi*50*t)
    #     i_values = 2*np.sqrt(2)*np.sin(2*np.pi*50*t+60*np.pi/180) # cos_phi = 0.5
    #     self.u_channel.put_data(u_values)
    #     self.i_channel.put_data(i_values)
    #     self.time_channel.put_data((t+abs_ts_start)*1e6)
    #     self.power_system.process()

    #     u_rms, sidx = self.power_system.output_channels["U1_rms"].read_data_by_acq_sidx(0, u_values.size)
    #     self.assertAlmostEqual(sidx[5*5],5.2*self.power_system._samplerate, places=-1)
    
class TestPowerSystemFluctuation(unittest.TestCase):
    def setUp(self):
        self.u_channel = AcqBuffer()
        self.time_channel = AcqBuffer(dtype=np.int64)

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u_channel,
            input_samplerate=5555.555,
            zcd_threshold=0.1
        )
        # Add Phase
        self.power_system.add_phase(u_channel=self.u_channel)
        self.power_system.enable_harmonic_calculation(num_harmonics=1)
        self.power_system.enable_nper_abs_time_sync(self.time_channel)

    def test_steady_state_60s(self):
        self.power_system.enable_fluctuation_calculation(nominal_voltage=230, pst_interval_sec=60)
        abs_ts_start = datetime.datetime(2024,1,1,0,0,59, tzinfo=datetime.UTC).timestamp()
        t = np.linspace(0, 81, int(self.power_system._samplerate*81), endpoint=False)
        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*50*t)

        blocksize = 1000
        for blk_idx in range(t.size // blocksize):
            self.u_channel.put_data(u_values[blk_idx*blocksize:(blk_idx+1)*blocksize])
            self.time_channel.put_data((t[blk_idx*blocksize:(blk_idx+1)*blocksize]+abs_ts_start)*1e6)
            self.power_system.process()
        self.assertAlmostEqual(self.power_system.output_channels["U1_pst"].last_sample_value, 0, places=1)
        self.assertEqual(self.power_system.output_channels["U1_pst"].last_sample_acq_sidx, np.round(self.power_system._samplerate*(61+0.02)))

def test_steady_state_600s(self):
        self.power_system.enable_fluctuation_calculation(nominal_voltage=230, pst_interval_sec=600)
        abs_ts_start = datetime.datetime(2024,1,1,0,9,59, tzinfo=datetime.UTC).timestamp()
        t = np.linspace(0, 621, int(self.power_system._samplerate*621), endpoint=False)
        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*50*t)

        blocksize = 1000
        for blk_idx in range(t.size // blocksize):
            self.u_channel.put_data(u_values[blk_idx*blocksize:(blk_idx+1)*blocksize])
            self.time_channel.put_data((t[blk_idx*blocksize:(blk_idx+1)*blocksize]+abs_ts_start)*1e6)
            self.power_system.process()
        self.assertAlmostEqual(self.power_system.output_channels["U1_pst"].last_sample_value, 0, places=1)
        self.assertEqual(self.power_system.output_channels["U1_pst"].last_sample_acq_sidx, np.round(self.power_system._samplerate*621))


if __name__ == "__main__":
    unittest.main()
