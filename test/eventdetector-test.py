import unittest
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pqopen.eventdetector import EventDetectorLevelLow, EventDetectorLevelHigh, EventController
from daqopen.channelbuffer import DataChannelBuffer, AcqBuffer

class TestEventDetector(unittest.TestCase):
    def setUp(self):
        ...

    def test_level_low(self):
        data_channel = DataChannelBuffer("event_test")
        acq_sidx = [0,  10, 20, 30, 40, 50, 60,  70,  80, 90]
        values =  [100, 90, 80, 70, 80, 90, 100, 100, 60, 100]
        data_channel.put_data_multi(acq_sidx, values)

        detector = EventDetectorLevelLow(95, 2, data_channel)
        events = detector.process(0, 50)
        self.assertEqual(events[0]["start_sidx"], 10)
        self.assertEqual(events[0]["stop_sidx"], None)
        self.assertEqual(events[0]["extrem_value"], 70)
        events = detector.process(50, 100)
        self.assertEqual(events[0]["start_sidx"], 10)
        self.assertEqual(events[0]["stop_sidx"], 60)
        self.assertEqual(events[0]["extrem_value"], 70)
        self.assertEqual(events[1]["start_sidx"], 80)
        self.assertEqual(events[1]["stop_sidx"], 90)
        self.assertEqual(events[1]["extrem_value"], 60)

    def test_level_high(self):
        data_channel = DataChannelBuffer("event_test")
        acq_sidx = [0,  10, 20, 30, 40, 50, 60,  70,  80, 90]
        values =  [100, 110, 120, 130, 120, 110, 100, 100, 140, 100]
        data_channel.put_data_multi(acq_sidx, values)

        detector = EventDetectorLevelHigh(105, 2, data_channel)
        events = detector.process(0, 30)
        self.assertEqual(events[0]["start_sidx"], 10)
        self.assertEqual(events[0]["stop_sidx"], None)
        self.assertEqual(events[0]["extrem_value"], 120)
        unfinished_event = events[0]
        events = detector.process(30, 50)
        events = detector.process(50, 100)
        self.assertEqual(events[0]["start_sidx"], 10)
        self.assertEqual(events[0]["stop_sidx"], 60)
        self.assertEqual(events[0]["extrem_value"], 130)
        self.assertEqual(events[0]["id"], unfinished_event["id"])
        self.assertEqual(events[1]["start_sidx"], 80)
        self.assertEqual(events[1]["stop_sidx"], 90)
        self.assertEqual(events[1]["extrem_value"], 140)         

class TestEventController(unittest.TestCase):
    def setUp(self):
            ...

    def test_simple(self):
        sample_rate = 1000
        t = np.arange(0, 0.1, 1/sample_rate)*1e6
        time_channel = AcqBuffer(dtype=np.int64)
        time_channel.put_data(t)
        
        data_channel_1 = DataChannelBuffer("data_channel_1")
        acq_sidx = [0,  10, 20, 30, 40, 50, 60,  70,  80, 90]
        values =  [100, 90, 80, 70, 80, 90, 100, 100, 60, 100]
        data_channel_1.put_data_multi(acq_sidx, values)
        data_channel_2 = DataChannelBuffer("data_channel_2")
        acq_sidx = [0,  10, 20, 30, 40, 50, 60,  70,  80, 90]
        values =  [100, 110, 120, 130, 120, 110, 100, 100, 140, 100]
        data_channel_2.put_data_multi(acq_sidx, values)

        detector_1 = EventDetectorLevelLow(95, 2, data_channel_1)
        detector_2 = EventDetectorLevelHigh(105, 2, data_channel_2)

        event_controller = EventController(time_channel, sample_rate)
        event_controller.PROCESSING_DELAY_SECONDS = 0
        event_controller.add_event_detector(detector_1)
        event_controller.add_event_detector(detector_2)

        events = event_controller.process()
        self.assertEqual(events[0].start_ts, 0.01)
        self.assertEqual(events[0].stop_ts, 0.06)
        self.assertEqual(events[0].extrem_value, 70)
        self.assertEqual(events[0].channel, "data_channel_1")
        self.assertEqual(events[0].type, "LEVEL_LOW")

        self.assertEqual(events[2].start_ts, 0.01)
        self.assertEqual(events[2].stop_ts, 0.06)
        self.assertEqual(events[2].extrem_value, 130)
        self.assertEqual(events[2].channel, "data_channel_2")
        self.assertEqual(events[2].type, "LEVEL_HIGH")

    def test_real_data(self):
        data = np.genfromtxt(SCRIPT_DIR+"/data_files/event_data_level_low.csv", delimiter=",", skip_header=1)
        data[:,0] *= 1e3 # convert ts ms to us
        sample_rate = 55555
        acq_sidx = ((data[:,0]-data[0,0])*(sample_rate/1e6)).astype(np.int64)
        acq_sidx -= acq_sidx[0]

        time_channel = AcqBuffer(dtype=np.int64)
        data_channel = DataChannelBuffer("data_channel")

        detector = EventDetectorLevelLow(208, 2, data_channel)
        event_controller = EventController(time_channel, sample_rate)
        event_controller.add_event_detector(detector)

        for calc_idx in range(int(data.shape[0]/3)-3):
            start_ts = data[calc_idx*3,0]
            stop_ts = data[(calc_idx+1)*3,0]
            t = np.arange(start_ts, stop_ts, 1e6/sample_rate, dtype=np.int64)
            time_channel.put_data(t)
            data_channel.put_data_multi(acq_sidx[calc_idx*3:(calc_idx+1)*3],
                                        data[calc_idx*3:(calc_idx+1)*3,1])
            events = event_controller.process()
            if events:
                if events[0].stop_sidx:
                    ...



if __name__ == '__main__':
    unittest.main()
