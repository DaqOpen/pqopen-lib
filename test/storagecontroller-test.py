import unittest
import os
import sys
import numpy as np
import time
from pathlib import Path
import paho.mqtt.client as mqtt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from daqopen.channelbuffer import AcqBuffer, DataChannelBuffer
from pqopen.storagecontroller import StorageController, StoragePlan, TestStorageEndpoint, CsvStorageEndpoint, HomeAssistantStorageEndpoint
from pqopen.eventdetector import EventController, EventDetectorLevelLow


class TestStorageController(unittest.TestCase):
    def setUp(self):
        self.time_channel = AcqBuffer(dtype=np.int64)
        self.scalar_channel = DataChannelBuffer("scalar1")
        self.array_channel = DataChannelBuffer("array1", sample_dimension=10)
        self.samplerate = 1000
        self.storage_controller = StorageController(self.time_channel, self.samplerate)

    def test_one_storageplan_agg(self):
        storage_endpoint = TestStorageEndpoint("Test", "1234")
        # Configure Storage Plan
        storage_plan = StoragePlan(storage_endpoint, 0, interval_seconds=1)
        storage_plan.add_channel(self.scalar_channel)
        self.storage_controller.add_storage_plan(storage_plan)

        self.time_channel.put_data(np.arange(0, 10, 1/self.samplerate)*1e6)
        self.scalar_channel.put_data_single(1, 5)
        self.scalar_channel.put_data_single(1000, 10) 
        self.scalar_channel.put_data_single(1999, 20) # in interval window 2
        self.scalar_channel.put_data_single(2000, 30) # will be included in interval window 2 (because next value after)
        self.scalar_channel.put_data_single(2100, 40)

        self.storage_controller.process()

        self.assertEqual(storage_endpoint._aggregated_data_list[0], {"data": {"scalar1": 7.5}, "timestamp_us": 1000_000, "interval_sec": 1})
        self.assertEqual(storage_endpoint._aggregated_data_list[1], {"data": {"scalar1": 25.0}, "timestamp_us": 2000_000, "interval_sec": 1})
        self.assertEqual(storage_endpoint._aggregated_data_list[2], {"data": {"scalar1": 40.0}, "timestamp_us": 3000_000, "interval_sec": 1})

    def test_one_storageplan_series(self):
        storage_endpoint = TestStorageEndpoint("Test", "1234")
        # Configure Storage Plan
        storage_plan = StoragePlan(storage_endpoint, 0, interval_seconds=0)
        storage_plan.add_channel(self.scalar_channel)
        self.storage_controller.add_storage_plan(storage_plan)

        self.time_channel.put_data(np.arange(0, 10_000_000, 1e6//self.samplerate))
        for i in range(100):
            self.scalar_channel.put_data_single(i*100, i)

        self.storage_controller.process()

        expected_data_list0 = {}
        expected_data_list0["scalar1"] = {"data": {}, "timestamps": {}}
        expected_data_list0["scalar1"]["data"] = np.arange(0,10, 1, dtype=np.float64).tolist()
        expected_data_list0["scalar1"]["timestamps"] = np.arange(0,1000000, 100000).tolist()

        self.assertEqual(storage_endpoint._data_series_list[0], expected_data_list0)

    def test_one_storageplan_events(self):
        storage_endpoint = TestStorageEndpoint("Test", "1234")
        # Configure Storage Plan
        storage_plan = StoragePlan(storage_endpoint, 0, interval_seconds=0, store_events=True)
        self.storage_controller.add_storage_plan(storage_plan)

        sample_rate = 1000
        t = np.arange(0, 0.1, 1/sample_rate)*1e6
        time_channel = AcqBuffer()
        time_channel.put_data(t)
        
        data_channel_1 = DataChannelBuffer("data_channel_1")
        acq_sidx = [0,  10, 20, 30, 40, 50, 60,  70,  80, 90]
        values =  [100, 90, 80, 70, 80, 90, 100, 100, 60, 100]
        data_channel_1.put_data_multi(acq_sidx, values)

        detector_1 = EventDetectorLevelLow(95, 2, data_channel_1)

        event_controller = EventController(time_channel, sample_rate)
        event_controller.PROCESSING_DELAY_SECONDS = 0
        event_controller.add_event_detector(detector_1)

        events = event_controller.process()

        self.storage_controller.process_events(events)

        self.assertEqual(storage_endpoint._event_list[0].start_ts, 0.01)

    def test_one_storageplan_series_slow(self):
        start_timestamp = int(1000000000*1e6)
        storage_endpoint = TestStorageEndpoint("Test", "1234")
        # Configure Storage Plan
        storage_plan = StoragePlan(storage_endpoint, start_timestamp, interval_seconds=0)
        storage_plan.add_channel(self.scalar_channel)
        self.storage_controller.add_storage_plan(storage_plan)
        time_data = np.arange(start_timestamp, start_timestamp+60_000_000, int(1e6*(1/self.samplerate)))
        for i in range(59):
            self.time_channel.put_data(time_data[i*self.samplerate:(i+1)*self.samplerate])
            self.scalar_channel.put_data_single(i*self.samplerate, i)
            self.storage_controller.process()

        expected_data_list0 = {}
        expected_data_list0["scalar1"] = {"data": {}, "timestamps": {}}
        expected_data_list0["scalar1"]["data"] = np.arange(0,1, 1, dtype=np.float64).tolist()
        expected_data_list0["scalar1"]["timestamps"] = time_data[:1*self.samplerate:self.samplerate].tolist()

        self.assertEqual(storage_endpoint._data_series_list[0], expected_data_list0)

class TestStorageEndpoints(unittest.TestCase):
    def setUp(self):
        self.time_channel = AcqBuffer(dtype=np.int64)
        self.scalar_channel = DataChannelBuffer("scalar1")
        self.array_channel = DataChannelBuffer("array1", sample_dimension=10)
        self.samplerate = 1000
        self.storage_controller = StorageController(self.time_channel, self.samplerate)

    def test_csv_endpoint(self):
        # Define Endpoint
        csv_endpoint = CsvStorageEndpoint(name="Test",
                                          measurement_id="1234",
                                          file_path=".")
        # Configure Storage Plan
        storage_plan = StoragePlan(csv_endpoint, 0, interval_seconds=1)
        storage_plan.add_channel(self.scalar_channel)
        storage_plan.add_channel(self.array_channel)
        self.storage_controller.add_storage_plan(storage_plan)

        self.time_channel.put_data(np.arange(0, 10_000_000, 1e6//self.samplerate))
        for i in range(100):
            self.scalar_channel.put_data_single(i*100, i)

        self.storage_controller.process()
        # Wait until finished!!!

    def test_ha_mqtt_endpoint(self):
        self.w_pos_channel = DataChannelBuffer("W_pos")
        self.w_neg_channel = DataChannelBuffer("W_neg")
        # Define Endpoint
        ha_mqtt_endpoint = HomeAssistantStorageEndpoint(
            name="Test",
            device_id="1234",
            mqtt_host="localhost",
            client_id="unittest")
        # Configure Storage Plan
        storage_plan = StoragePlan(ha_mqtt_endpoint, 0, interval_seconds=1)
        storage_plan.add_channel(self.w_pos_channel)
        storage_plan.add_channel(self.w_neg_channel)
        self.storage_controller.add_storage_plan(storage_plan)

        self.time_channel.put_data(np.arange(0, 10_000_000, 1e6//self.samplerate))
        for i in range(100):
            self.w_pos_channel.put_data_single(i*100, i)
            self.w_neg_channel.put_data_single(i*100, i)

        self.storage_controller.process()

        time.sleep(2)
        # Wait until finished!!!


if __name__ == "__main__":
    unittest.main()



