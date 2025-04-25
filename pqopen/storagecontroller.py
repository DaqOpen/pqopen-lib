# pqopen/storagecontroller.py

"""
The `pqopen/storagecontroller.py` module provides a flexible framework for managing data storage across multiple endpoints. 
It supports various storage methods, including CSV files, MQTT (for IoT integration), and Home Assistant-specific MQTT topics.

The module contains the following key components:

- **StorageEndpoint**: An abstract base class defining the interface for writing data to different storage endpoints.
- **StoragePlan**: Manages the configuration for storing data from specific channels at specified intervals.
- **StorageController**: Orchestrates the processing and storage of data across all configured plans.

# Features

- Supports multiple storage endpoints (e.g., CSV, MQTT, Home Assistant).
- Handles both raw time-series data and aggregated data.
- Enables event-driven data storage.
-   Designed for flexibility in integrating different storage methods.

# Usage

The module is designed to be used in applications requiring robust data storage solutions. 
It is particularly useful for power quality applications, real-time monitoring systems, 
and scenarios where data needs to be stored locally and/or transmitted to remote services.

Examples:

```python
endpoints = {
"csv": {"data_dir": "/tmp/"},
"mqtt": {
    "hostname": "localhost",
    "client_id": "pqopen-mqtt",
    "topic_prefix": "dt/pqopen"
    }
}
storage_plans = {
"plan1": {
    "endpoint": "csv",
    "interval_sec": 600,
    "channels": ["channel1", "channel2"]
    },
"plan2": {
    "endpoint": "mqtt",
    "interval_sec": 10,
    "store_events": True
    }
}
# Initialize the StorageController and configure endpoints and plans
storage_controller = StorageController(time_channel, sample_rate)
storage_controller.setup_endpoints_and_storageplans(endpoints=endpoints, storage_plans=storage_plans, available_channels=channels, measurement_id="unique_measurement_id", device_id="device_123", start_timestamp_us=start_ts)
```
For detailed usage and implementation details, refer to the individual class documentation within the module.
"""

import numpy as np
from pqopen.helper import floor_timestamp, JsonDecimalLimiter
from pqopen.eventdetector import Event
from daqopen.channelbuffer import DataChannelBuffer, AcqBuffer
from pathlib import Path
from typing import List, Dict
import logging
import json
import gzip
import paho.mqtt.client as mqtt


logger = logging.getLogger(__name__)

class StorageEndpoint(object):
    """Represents an endpoint for storing data."""

    def __init__(self, name: str, measurement_id: str):
        """
        Parameters:
            name: The name of the storage endpoint.
        """
        self.name = name
        self.measurement_id = measurement_id

    def write_data_series(self, data: dict):
        """
        Writes a series of data to the storage endpoint.

        Parameters:
            data: The data to be stored, organized by channels.
        """
        pass

    def write_aggregated_data(self, data: dict, timestamp_us: int, interval_seconds: int):
        """
        Writes aggregated data to the storage endpoint.

        Args:
            data: The aggregated data to store.
            timestamp_us: The timestamp in microseconds for the aggregated data.
            interval_seconds: The aggregation interval in seconds.
        """
        pass

    def write_event(self, event: Event):
        pass

class StoragePlan(object):
    """Defines a plan for storing data with specified intervals and channels."""

    def __init__(self, 
                 storage_endpoint: StorageEndpoint, 
                 start_timestamp_us: int, 
                 interval_seconds: int=10, 
                 storage_name: str='aggregated_data', 
                 store_events: bool=False):
        """
        Parameters:
            storage_endpoint: The storage endpoint to use.
            start_timestamp_us: Starting timestamp in µs
            interval_seconds: The interval for aggregation in seconds.
            storage_name: Name of the storage dataset.
            store_events: Flag if events should be also stored or not
        """
        self.storage_endpoint = storage_endpoint
        self.interval_seconds = interval_seconds
        self.channels: List[Dict] = []
    
        self._storage_counter = 0
        self._store_events_enabled = store_events
        self.next_storage_timestamp = start_timestamp_us
        self.next_storage_sample_index = 0
        self.last_storage_sample_index = 0

        self.storage_name = storage_name

    def add_channel(self, channel: DataChannelBuffer):
        """
        Adds a data channel to the storage plan.

        Parameters:
            channel: The channel to add.
        """
        self.channels.append({"channel": channel, "last_store_sidx": 0})

    def store_data_series(self, time_channel: AcqBuffer, sample_rate: float):
        """
        Stores a series of data (1:1) from the channels in the storage plan.

        Parameters:
            time_channel: The time channel for converting the acq_sidx to real timestamps.
        """
        data = {}
        for channel in self.channels:
            channel_timestamps = []
            channel_sample_indices = []
            if isinstance(channel["channel"], DataChannelBuffer):
                channel_data, channel_sample_indices = channel["channel"].read_data_by_acq_sidx(self.last_storage_sample_index, self.next_storage_sample_index)
                # Convert to serializable data types
                channel_data = channel_data.tolist()
                channel_sample_indices = channel_sample_indices.tolist()
            else:
                logger.warning("Channel is not of instance DataChannelBuffer")

            if channel_sample_indices:
                for sample_index in channel_sample_indices:
                    channel_timestamps.append(int(time_channel.read_data_by_index(sample_index, sample_index + 1)[0]))
                data[channel["channel"].name] = {'data': channel_data, 'timestamps': channel_timestamps}
        if data:
            self.storage_endpoint.write_data_series(data)

    def store_aggregated_data(self, stop_sidx: int):
        """
        Stores aggregated data from the channels in the storage plan.

        Parameters:
            stop_sidx: The stopping sample index for aggregation.
        """
        data = {}
        # TODO: Only incude next if timestamp is not round
        for channel in self.channels:
            channel_data, last_included_sidx = channel["channel"].read_agg_data_by_acq_sidx(
                channel["last_store_sidx"], stop_sidx, include_next=True
            )
            data[channel["channel"].name] = channel_data
            channel["last_store_sidx"] = last_included_sidx+1 if last_included_sidx else stop_sidx

        self.storage_endpoint.write_aggregated_data(data, self.next_storage_timestamp, self.interval_seconds)
        #self.last_storage_sample_index = last_included_sidx+1 if last_included_sidx else stop_sidx

    def store_event(self, event: Event):
        if self._store_events_enabled:
            self.storage_endpoint.write_event(event)

class StorageController(object):
    """Manages multiple storage plans and processes data for storage."""

    STORAGE_DELAY_SECONDS = 1
    DATA_SERIES_PACKET_TIME = int(1e6)

    def __init__(self, time_channel: AcqBuffer, sample_rate: float):
        """
        Parameters:
            time_channel: The acquisition buffer for timestamps.
            sample_rate: The sampling rate in Hz. 
        """
        self.time_channel = time_channel
        self.sample_rate = sample_rate
        self.storage_plans: List[StoragePlan] = []
        self._last_processed_sidx = 0
        self._last_processed_sidx = 0
        self._unfinished_event_ids = []

    def add_storage_plan(self, storage_plan: StoragePlan):
        """
        Adds a storage plan to the controller.

        Parameters:
            storage_plan: The storage plan to add.
        """
        self.storage_plans.append(storage_plan)

    def process(self):
        """
        Processes data for all storage plans based on the current acquisition state.
        """
        start_acq_sidx = self._last_processed_sidx
        stop_acq_sidx = self.time_channel.sample_count - int(self.STORAGE_DELAY_SECONDS*self.sample_rate)
        if stop_acq_sidx <= 0:
            return None

        timestamps = self.time_channel.read_data_by_index(start_acq_sidx, stop_acq_sidx)
        
        for storage_plan in self.storage_plans:
            if storage_plan.interval_seconds <= 0:
                self._process_data_series(storage_plan, start_acq_sidx, timestamps)
            else:
                self._process_aggregated_data(storage_plan, start_acq_sidx, timestamps)
                
        self._last_processed_sidx = stop_acq_sidx

    def _process_data_series(self, storage_plan: StoragePlan, start_acq_sidx: int, timestamps: np.ndarray):
        """
        Processes data series for a specific storage plan.

        Parameters:
            storage_plan: The storage plan to process.
            start_acq_sidx: The starting sample index.
            timestamps: The array of timestamps.
        """
        while storage_plan.next_storage_timestamp <= timestamps.max():
            #if timestamps.min() < storage_plan.next_storage_timestamp:
            storage_plan.next_storage_sample_index = start_acq_sidx + timestamps.searchsorted(storage_plan.next_storage_timestamp)
            storage_plan.store_data_series(self.time_channel, self.sample_rate)
            storage_plan.last_storage_sample_index = storage_plan.next_storage_sample_index
            storage_plan.next_storage_timestamp += self.DATA_SERIES_PACKET_TIME

    def _process_aggregated_data(self, storage_plan: StoragePlan, start_acq_sidx: int, timestamps: np.ndarray):
        """
        Processes aggregated data for a specific storage plan.

        Args:
            storage_plan: The storage plan to process.
            start_acq_sidx: The starting sample index.
            timestamps: The array of timestamps.
        """
        while storage_plan.next_storage_timestamp <= timestamps.max():
            # Check if storage plan timestamp is in the current time span (-1 Sample)
            if storage_plan._storage_counter and ((timestamps.min() - int(1e6/self.sample_rate)) < storage_plan.next_storage_timestamp):
                stop_store_sidx = start_acq_sidx + timestamps.searchsorted(storage_plan.next_storage_timestamp)
                storage_plan.store_aggregated_data(stop_store_sidx)
                logger.debug(f"Storage Plan {storage_plan.storage_name}: stop_store_sidx={stop_store_sidx:d} next_storage_timestamp={storage_plan.next_storage_timestamp:d} ts_min={timestamps.min():d} ts_max={timestamps.max():d}")
            else:
                logger.debug(f"Storage Plan {storage_plan.storage_name}: next_storage_timestamp={storage_plan.next_storage_timestamp:d} ts_min={timestamps.min():d} ts_max={timestamps.max():d}")
            # Calculate next round timestamp for storing
            storage_plan.next_storage_timestamp = int(floor_timestamp(timestamp=storage_plan.next_storage_timestamp + int(storage_plan.interval_seconds*1e6),
                                                                      interval_seconds=storage_plan.interval_seconds,
                                                                      ts_resolution="us"))
            storage_plan._storage_counter += 1

    def process_events(self, events: List[Event]):
        """
        Process events to be stored

        Parameters:
            events: List of events to be stored by each storage plan
        """
        for event in events:
            if event.stop_ts is None:
                if event.id in self._unfinished_event_ids:
                    continue # ignore already known unfinished events
                else:
                    self._unfinished_event_ids.append(event.id)
                    logger.debug(f"add event_id {str(event.id)} to unfinished events")
            else:
                logger.debug(f"finished event")
                if event.id in self._unfinished_event_ids:
                    logger.debug(f"remove event_id {str(event.id)} from unfinished events")
                    self._unfinished_event_ids.remove(event.id)
            for storage_plan in self.storage_plans:
                storage_plan.store_event(event)
            
    def setup_endpoints_and_storageplans(self, 
                                         endpoints: dict, 
                                         storage_plans: dict, 
                                         available_channels: dict, 
                                         measurement_id: str, 
                                         device_id: str,
                                         start_timestamp_us: int):
        """
        Setup endpoints and storage plans from config

        Args:
            endpoints: Dict of endpoints to be configured (csv and persistmq are supported for now)
            storage_plans: Dict of storageplans to be created
            available_channels: List of all available channels
            measurement_id: The actual measurement id for tagging the session
            device_id: Id of the device for unique tagging the data origin
            start_timestamp_us: Timestamp of the start of measurmement

        Raises:
            NotImplementedError: If a not implemented endpoint will be configured
        """
        self._configured_eps = {}
        for ep_type, ep_config in endpoints.items():
            if ep_type == "csv":
                csv_storage_endpoint = CsvStorageEndpoint("csv", measurement_id, ep_config.get("data_dir", "/tmp/"))
                self._configured_eps["csv"] = csv_storage_endpoint
            elif ep_type == "mqtt":
                mqtt_storage_endpoint = MqttStorageEndpoint(name="mqtt", 
                                                            measurement_id=measurement_id, 
                                                            device_id=device_id, 
                                                            mqtt_host=ep_config.get("hostname", "localhost"), 
                                                            client_id=ep_config.get("client_id", "pqopen-mqtt"),
                                                            topic_prefix=ep_config.get("topic_prefix", "dt/pqopen"),
                                                            compression=ep_config.get("compression", False),
                                                            limit_precision=ep_config.get("decimal_places", None))
                self._configured_eps["mqtt"] = mqtt_storage_endpoint
            elif ep_type == "ha_mqtt":
                ha_mqtt_storage_endpoint = HomeAssistantStorageEndpoint(name="ha_mqtt",
                                                            device_id=device_id, 
                                                            mqtt_host=ep_config.get("hostname", "localhost"), 
                                                            client_id=ep_config.get("client_id", "pqopen-ha"),
                                                            topic_prefix=ep_config.get("topic_prefix", "pqopen/data"),
                                                            qos=ep_config.get("qos", 0))
                self._configured_eps["ha_mqtt"] = ha_mqtt_storage_endpoint
            else:
                raise NotImplementedError(f"{ep_type:s} not implemented")
        for sp_name, sp_config in storage_plans.items():
            sp_endpoint = sp_config.get("endpoint")
            if sp_endpoint not in self._configured_eps:
                logger.warning(f"Endpoint {sp_endpoint:s} not configured")
                continue
            storage_plan = StoragePlan(storage_endpoint=self._configured_eps[sp_endpoint],
                                       start_timestamp_us=start_timestamp_us,
                                       interval_seconds=sp_config.get("interval_sec", 600),
                                       storage_name=sp_name,
                                       store_events=sp_config.get("store_events", False))
            channels_to_store = sp_config.get("channels", [])
            if not channels_to_store:
                # Add all available channels
                for channel in available_channels.values():
                    storage_plan.add_channel(channel)
            else:
                for channel in channels_to_store:
                    if channel in available_channels:
                        if len(available_channels[channel]._data.shape) == 1:
                            storage_plan.add_channel(available_channels[channel])
                        else:
                            logger.warning(f"Channel {channel} not a scalar")
                    else:      
                        logger.warning(f"Channel {channel} not available for storing")
            self.add_storage_plan(storage_plan)

            

class TestStorageEndpoint(StorageEndpoint):
    """A implementation of StorageEndpoint for testing purposes."""

    def __init__(self, name, measurement_id):
        super().__init__(name, measurement_id)
        self._data_series_list = []
        self._aggregated_data_list = []
        self._event_list = []

    def write_data_series(self, data):
        self._data_series_list.append(data)

    def write_aggregated_data(self, data, timestamp_us, interval_seconds):
        self._aggregated_data_list.append({"data": data, "timestamp_us": timestamp_us, "interval_sec": interval_seconds})

    def write_event(self, event):
        self._event_list.append(event)


class MqttStorageEndpoint(StorageEndpoint):
    """Represents a MQTT endpoint (MQTT) for transferring data."""
    def __init__(self, 
                 name: str, 
                 measurement_id: str, 
                 device_id: str, 
                 mqtt_host: str, 
                 client_id: str, 
                 topic_prefix: str = "dt/pqopen",
                 compression: bool=True,
                 limit_precision: int = None):
        """ Create a MQTT storage endpoint

        Parameters:
            name: The name of the endpoint
            measurement_id: Id of the measurement, will be indcluded in the transmitted data
            device_id: The device Id
            mqtt_host: hostname of the MQTT broker.
            client_id: name to be used for mqtt client identification
            topic_prefix: topic prefix before device-id, no trailing /
            compression: Flag if payload should be compressed with gzip or not
            limit_precision: Limit float precision to number of decimal places
        """
        super().__init__(name, measurement_id)
        self._device_id = device_id
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)
        self._client.connect_async(host=mqtt_host)
        self._compression = compression
        self._precision_limiter = JsonDecimalLimiter(limit_precision) if limit_precision is not None else None
        self._topic_prefix = topic_prefix
        self._client.loop_start()

    def write_aggregated_data(self, data: dict, timestamp_us: int, interval_seconds: int):
        """ Write an aggregated data message

        Parameters:
            data: The data object to be sent
            timestamp_us: Timestamp (in µs) of the data set
            interval_seconds: Aggregation intervall, used as data tag
        """
        agg_data_obj = {'type': 'aggregated_data',
                        'measurement_uuid': self.measurement_id,
                        'interval_sec': interval_seconds,
                        'timestamp': timestamp_us/1e6,
                        'data': data}
        json_item = json.dumps(agg_data_obj)
        if self._precision_limiter:
            json_item = self._precision_limiter.process(json_item)
        if self._compression:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/agg_data/gjson",
                            gzip.compress(json_item.encode('utf-8')), qos=2)
        else:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/agg_data/json",
                            json_item.encode('utf-8'), qos=2)
            
    def write_data_series(self, data: dict):
        """ Write a timeseries data message

        Parameters:
            data: The data object to be sent
        """
        data_series_obj = {'type': 'timeseries_data',
                        'measurement_uuid': self.measurement_id,
                        'data': data}
        json_item = json.dumps(data_series_obj)
        if self._precision_limiter:
            json_item = self._precision_limiter.process(json_item)
        if self._compression:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/dataseries/gjson",
                            gzip.compress(json_item.encode('utf-8')), qos=2)
        else:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/dataseries/json",
                            json_item.encode('utf-8'), qos=2)
            
    def write_event(self, event: Event):
        """
        Write event data message

        Parameters:
            event: The event to be written
        """
        event_obj = {
            "type": "event",
            "measurement_uuid": self.measurement_id,
            "event_type": event.type,
            "timestamp": event.start_ts,
            "channel": event.channel,
            "data": {"duration": (event.stop_ts - event.start_ts) if event.stop_ts else None,
                     "extrem_value": event.extrem_value,
                     "id": str(event.id)}
        }
        json_item = json.dumps(event_obj)
        if self._compression:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/event/gjson",
                            gzip.compress(json_item.encode('utf-8')), qos=2)
        else:
            self._client.publish(self._topic_prefix + f"/{self._device_id:s}/event/json",
                            json_item.encode('utf-8'), qos=2)
            
class HomeAssistantStorageEndpoint(StorageEndpoint):
    """Represents a MQTT endpoint (MQTT) for HomeAssistant."""
    def __init__(self, 
                 name: str, 
                 device_id: str, 
                 mqtt_host: str, 
                 client_id: str, 
                 topic_prefix: str = "pqopen/data",
                 qos: int = 0):
        """ Create an endpoint for HomeAssistant via MQTT

        Parameters:
            name: The name of the endpoint
            device_id: The device Id
            mqtt_host: hostname of the MQTT broker.
            client_id: name to be used for mqtt client identification
            topic_prefix: topic prefix before device-id, no trailing /
            qos: mqtt quality of service. valid values 0, 1, 2
        """
        super().__init__(name, measurement_id=None)
        self._device_id = device_id
        self._topic_prefix = topic_prefix
        self._qos = qos
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.connect_async(host=mqtt_host)
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
            logger.debug("HA-MQTT Connected")
            homeassistant_config = {"device_class": "power",
                                    "name": "Active Power Positive",
                                    "unique_id": "pqopen_P_pos",
                                    "state_topic": f"{self._topic_prefix:s}",
                                    "force_update": True,
                                    "unit_of_measurement": "W",
                                    "value_template": "{{ value_json.P_pos}}"}
            client.publish("homeassistant/sensor/activePowerPositive/config", json.dumps(homeassistant_config), retain=True)

            homeassistant_config = {"device_class": "power",
                                "name": "Active Power Negative",
                                "unique_id": "pqopen_P_neg",
                                "state_topic": f"{self._topic_prefix:s}",
                                "force_update": True,
                                "unit_of_measurement": "W",
                                "value_template": "{{ value_json.P_neg}}"}
            client.publish("homeassistant/sensor/activePowerNegative/config", json.dumps(homeassistant_config), retain=True)

            homeassistant_config = {"device_class": "energy",
                                    "state_class": "total",
                                    "name": "Active Energy Positive",
                                    "unique_id": "pqopen_W_pos",
                                    "state_topic": f"{self._topic_prefix:s}",
                                    "force_update": True,
                                    "unit_of_measurement": "Wh",
                                    "value_template": "{{ value_json.W_pos}}"}
            client.publish("homeassistant/sensor/activeEnergyPositive/config", json.dumps(homeassistant_config), retain=True)

            homeassistant_config = {"device_class": "energy",
                                    "state_class": "total",
                                "name": "Active Energy Negative",
                                "unique_id": "pqopen_W_neg",
                                "state_topic": f"{self._topic_prefix:s}",
                                "force_update": True,
                                "unit_of_measurement": "Wh",
                                "value_template": "{{ value_json.W_neg}}"}
            client.publish("homeassistant/sensor/activeEnergyNegative/config", json.dumps(homeassistant_config), retain=True)

    def write_aggregated_data(self, data: dict, timestamp_us: int, interval_seconds: int):
        """ Write an aggregated data message

        Parameters:
            data: The data object to be sent
            timestamp_us: Timestamp (in µs) of the data set
            interval_seconds: Aggregation intervall, used as data tag
        """
        self._client.publish(self._topic_prefix, json.dumps(data).encode("utf-8"), qos=self._qos)
        logger.debug("Published HA-MQTT Message")
            
    

class CsvStorageEndpoint(StorageEndpoint):
    """Represents a csv storage endpoint"""
    def __init__(self, name: str, measurement_id: str, file_path: str | Path):
        """ Create a csv storage endpoint

        Parameters:
            name: The name of the endpoint
            measurement_id: Id of the measurement, will be indcluded in the transmitted data
            file_path: Data path for the csv file
        """
        super().__init__(name, measurement_id)
        self._file_path = file_path
        self._header_keys = []
        self._file_path = file_path if isinstance(file_path, Path) else Path(file_path)
        self._file_path.mkdir(parents=True, exist_ok=True)

    def write_aggregated_data(self, data: dict, timestamp_us: int, interval_seconds: int):
        # Filter scalar values for now
        data = {key: value for key, value in data.items() if isinstance(value, (float, int, type(None)))}
        file_path = self._file_path/f"{self.measurement_id}_{interval_seconds:d}s.csv"
        channel_names = list(data.keys())
        if not self._header_keys:
            self._header_keys = channel_names
            file_path.write_text("timestamp," + ",".join(channel_names)+"\n")
        if self._header_keys != channel_names:
            logger.warning("CSV-Writer: Channel names and known keys differ!")
        with open(file_path, "a") as f:
            f.write(f"{timestamp_us/1e6:.3f},")
            f.write(",".join([f"{data[key]:.3f}" if isinstance(data[key], (float, int)) else "" for key in self._header_keys])+"\n")