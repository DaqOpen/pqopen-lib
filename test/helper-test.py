import unittest
import sys
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pqopen.helper import floor_timestamp

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


if __name__ == '__main__':
    unittest.main()