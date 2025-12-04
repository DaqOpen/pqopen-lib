import unittest
import sys
import os
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pqopen.helper import floor_timestamp, JsonDecimalLimiter, create_harm_corr_array, create_fft_corr_array

class TestFloorTimestamp(unittest.TestCase):


    def test_simple_10s_interval(self):
        """
        Test floor of 10s interval.
        """
        ts = datetime(2024,11,22,8,13,4)
        interval = 10 # seconds
        expected_floor_ts = datetime(2024,11,22,8,13).timestamp()
        
        floored_ts = floor_timestamp(ts.timestamp(), interval, ts_resolution="s")
        self.assertEqual(floored_ts, expected_floor_ts)

        ts_ms = int(ts.timestamp()*1_000)
        floored_ts = floor_timestamp(ts_ms, interval, ts_resolution="ms")
        self.assertEqual(floored_ts, expected_floor_ts*1_000)

        ts_us = int(ts.timestamp()*1_000_000)
        floored_ts = floor_timestamp(ts_us, interval, ts_resolution="us")
        self.assertEqual(floored_ts, expected_floor_ts*1_000_000)


    def test_simple_10min_interval(self):
        """
        Test floor of 10min interval.
        """
        ts = datetime(2024,11,22,8,13,4)
        interval = 600 # seconds
        expected_floor_ts = datetime(2024,11,22,8,10).timestamp()
        
        floored_ts = floor_timestamp(ts.timestamp(), interval, ts_resolution="s")
        self.assertEqual(floored_ts, expected_floor_ts)

        ts_ms = int(ts.timestamp()*1_000)
        floored_ts = floor_timestamp(ts_ms, interval, ts_resolution="ms")
        self.assertEqual(floored_ts, expected_floor_ts*1_000)

        ts_us = int(ts.timestamp()*1_000_000)
        floored_ts = floor_timestamp(ts_us, interval, ts_resolution="us")
        self.assertEqual(floored_ts, expected_floor_ts*1_000_000)

        ts_us = 1_733_505_000_010_123
        next_round_ts_us = int(floor_timestamp(ts_us, 600, ts_resolution="us")+600*1e6)
        self.assertEqual(next_round_ts_us, 1_733_505_600_000_000)

class TestLimitDecimalPlaces(unittest.TestCase):
   
    def test_basic_rounding(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"value": 3.14159}'
        expected_output = '{"value": 3.14}'
        self.assertEqual(float_limiter.process(input_json), expected_output)

    def test_multiple_floats(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"a": 1.2345, "b": 6.78901}'
        expected_output = '{"a": 1.23, "b": 6.79}'
        self.assertEqual(float_limiter.process(input_json), expected_output)

    def test_no_change_needed(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"a": 1.20, "b": 2.00}'
        self.assertEqual(float_limiter.process(input_json), input_json)

    def test_negative_numbers(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"neg": -2.71828}'
        expected_output = '{"neg": -2.72}'
        self.assertEqual(float_limiter.process(input_json), expected_output)

    def test_string_numbers_unchanged(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"price": "3.14159"}'
        self.assertEqual(float_limiter.process(input_json), input_json)

    def test_integer_values(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"int": 5}'
        self.assertEqual(float_limiter.process(input_json), input_json)

    def test_custom_decimal_places(self):
        float_limiter = JsonDecimalLimiter(3)
        input_json = '{"value": 3.14159}'
        expected_output = '{"value": 3.142}'
        self.assertEqual(float_limiter.process(input_json), expected_output)

    def test_scientific_notation_ignored(self):
        float_limiter = JsonDecimalLimiter()
        input_json = '{"sci": 1.23e10}'
        self.assertEqual(float_limiter.process(input_json), input_json)

class TestHarmCorrCreater(unittest.TestCase):

    def test_harmonic(self):
        freq_response = ((50, 1.0), (100, 0.9))
        expected_corr_factors = 1/np.array([1, 1, 0.9, 0.9, 0.9, 0.9])
        corr_factors = create_harm_corr_array(50, 5, freq_response)

        self.assertIsNone(np.testing.assert_array_almost_equal(corr_factors, expected_corr_factors))

    def test_interharmonic(self):
        freq_response = ((50, 1.0), (100, 0.9))
        expected_corr_factors = 1/np.array([1, 0.95, 0.9, 0.9, 0.9, 0.9])
        corr_factors = create_harm_corr_array(50, 5, freq_response, interharm=True)

        self.assertIsNone(np.testing.assert_array_almost_equal(corr_factors, expected_corr_factors))

class TestFftCorrCreater(unittest.TestCase):

    def test_small(self):
        freq_response = ((50, 1.0), (100, 0.9))

        expected_corr_factors = 1/np.array([1, 1, 0.9, 0.9, 0.9, 0.9])
        corr_factors = create_fft_corr_array(6, 250, freq_response)

        self.assertIsNone(np.testing.assert_array_almost_equal(corr_factors, expected_corr_factors))

if __name__ == '__main__':
    unittest.main()