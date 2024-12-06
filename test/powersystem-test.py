import unittest
import os
import sys
import numpy as np
import datetime
from unittest.mock import MagicMock

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
            input_samplerate=1000.0,
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
    
class TestPowerSystemFluctuation(unittest.TestCase):
    def setUp(self):
        self.u_channel = AcqBuffer()
        self.time_channel = AcqBuffer(dtype=np.int64)

        # Create PowerSystem instance
        self.power_system = PowerSystem(
            zcd_channel=self.u_channel,
            input_samplerate=5000.0,
            zcd_threshold=0.1
        )
        # Add Phase
        self.power_system.add_phase(u_channel=self.u_channel)
        self.power_system.enable_harmonic_calculation(num_harmonics=1)
        self.power_system.enable_nper_abs_time_sync(self.time_channel)
        self.power_system.enable_fluctuation_calculation(nominal_voltage=230, pst_interval_sec=60)

    def test_steady_state(self):
        abs_ts_start = datetime.datetime(2024,1,1,0,0,59, tzinfo=datetime.UTC).timestamp()
        t = np.linspace(0, 81, int(self.power_system._samplerate)*81, endpoint=False)
        u_values = 230*np.sqrt(2)*np.sin(2*np.pi*50*t)

        blocksize = 1000
        for blk_idx in range(t.size // blocksize):
            self.u_channel.put_data(u_values[blk_idx*blocksize:(blk_idx+1)*blocksize])
            self.time_channel.put_data((t[blk_idx*blocksize:(blk_idx+1)*blocksize]+abs_ts_start)*1e6)
            self.power_system.process()
        self.assertAlmostEqual(self.power_system.output_channels["U1_pst"].last_sample_value, 0, places=1)
        self.assertEqual(self.power_system.output_channels["U1_pst"].last_sample_acq_sidx, self.power_system._samplerate*61)


if __name__ == "__main__":
    unittest.main()
