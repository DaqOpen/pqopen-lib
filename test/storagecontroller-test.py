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
from pqopen.storagecontroller import StorageController, StoragePlan, TestStorageEndpoint, DaqOpenServerStorageEndpoint


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
        expected_data_list0["scalar1"]["data"] = np.arange(0,50, 1, dtype=np.float64).tolist()
        expected_data_list0["scalar1"]["timestamps"] = np.arange(0,5000000, 100000).tolist()

        self.assertEqual(storage_endpoint._data_series_list[0], expected_data_list0)

class TestStorageEndpoints(unittest.TestCase):
    def setUp(self):
        self.time_channel = AcqBuffer(dtype=np.int64)
        self.scalar_channel = DataChannelBuffer("scalar1")
        self.array_channel = DataChannelBuffer("array1", sample_dimension=10)
        self.samplerate = 1000
        self.storage_controller = StorageController(self.time_channel, self.samplerate)

    def test_persist_mq_endpoint(self):
        # Define Endpoint
        test_endpoint = DaqOpenServerStorageEndpoint(name="Test",
                                                     measurement_id="1234",
                                                     device_id="b990cdff-87fb-4f82-b443-864ce7ba1731",
                                                     mqtt_host="localhost",
                                                     client_id="testclient",
                                                     cache_path=Path("/tmp/"))
        # Configure Storage Plan
        storage_plan = StoragePlan(test_endpoint, 0, interval_seconds=1)
        storage_plan.add_channel(self.scalar_channel)
        self.storage_controller.add_storage_plan(storage_plan)

        self.time_channel.put_data(np.arange(0, 10_000_000, 1e6//self.samplerate))
        for i in range(100):
            self.scalar_channel.put_data_single(i*100, i)

        self.storage_controller.process()
        # Wait until finished!!!
        time.sleep(1)  


if __name__ == "__main__":
    unittest.main()



