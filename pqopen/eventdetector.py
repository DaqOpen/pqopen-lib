"""
DAQ-based Event Detection Module

This module provides classes for detecting events in data acquisition (DAQ) systems. It includes:

- A `Event` dataclass to represent detected events
- Base `EventDetector` class for event detection logic
- Subclasses `EventDetectorLevelLow` and `EventDetectorLevelHigh` for specific event detection criteria
- An `EventController` class to manage multiple event detectors and process events

Classes:
    Event: A dataclass representing a detected event with associated metadata.
    EventDetector: Base class for implementing various event detection algorithms.
        EventDetectorLevelLow: Detects events where data drops below a specified limit.
        EventDetectorLevelHigh: Detects events where data exceeds a specified limit.
    EventController: Manages multiple event detectors and processes events over time.
"""

import numpy as np
from daqopen.channelbuffer import DataChannelBuffer, AcqBuffer
from pathlib import Path
from typing import List, Dict
import logging
import json
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Represents a detected event with associated metadata.

    Attributes:
        start_ts: Timestamp (in seconds) when the event started.
        stop_ts: Timestamp (in seconds) when the event stopped. None if ongoing.
        start_sidx: Start sample index of the event.
        stop_sidx: Stop sample index of the event. None if ongoing.
        extrem_value: Extreme value associated with the event (minimum for Level Low, maximum for Level High).
        channel: Name of the channel from which the event was detected.
        type: Type of event ('LEVEL_LOW' or 'LEVEL_HIGH').
        id: Unique identifier for the event.
    """
    start_ts: float
    stop_ts: float
    start_sidx: int
    stop_sidx: int
    extrem_value: float
    channel: str
    type: str
    id: uuid.UUID


class EventDetector(object):
    """Base class for implementing event detection algorithms.

    Attributes:
        limit: The threshold value that defines the event boundary.
        threshold: The buffer around the limit used to detect event completion.
        observed_channel: The channel being monitored for events.
        last_channel_data: The most recent data from the observed channel.
        last_channel_sidx: The most recent sample index from the observed channel.
        _unfinished_event: Metadata for an event that is still ongoing.
        _type: Type of event detection algorithm ('LEVEL_LOW' or 'LEVEL_HIGH').
    """
    def __init__(self, limit: float, threshold: float, observed_channel: DataChannelBuffer):
        """Initializes the EventDetector with specified parameters.

        Args:
            limit: The threshold value that defines the event boundary.
            threshold: The buffer around the limit used to detect event completion.
            observed_channel: The channel being monitored for events.
        """
        self.limit = limit
        self.threshold = threshold
        self.observed_channel = observed_channel
        self.last_channel_data = None
        self.last_channel_sidx = None
        self._unfinished_event = {}
        self._type = ""

    def process(self, start_sidx: int, stop_sidx: int):
        ...

    def _get_channel_data(self, start_sidx, stop_sidx):
        data, sidx = self.observed_channel.read_data_by_acq_sidx(start_sidx, stop_sidx)
        if data.size < 1:
            return None, None
        if self.last_channel_data:
            data = np.r_[self.last_channel_data, data]
            sidx = np.r_[self.last_channel_sidx, sidx]
        self.last_channel_data = data[-1]
        self.last_channel_sidx = sidx[-1]
        return data, sidx

class EventDetectorLevelLow(EventDetector):
    """Detects events where data drops below a specified limit (LEVEL_LOW).

    Inherits from EventDetector and overrides the _type to 'LEVEL_LOW'.
    """
    def __init__(self, limit: float, threshold: float, observed_channel: DataChannelBuffer):
        """Initializes the Level Low event detector.

        Args:
            limit: The lower threshold value.
            threshold: The buffer below the limit used to detect event completion.
            observed_channel: The channel being monitored for events.
        """
        super().__init__(limit, threshold, observed_channel)
        self._type = "LEVEL_LOW"

    def process(self, start_sidx, stop_sidx):
        data, sidx = self._get_channel_data(start_sidx, stop_sidx)
        if data is None:
            return None
        limit_cross = np.diff(np.sign(data - self.limit))
        limit_cross_idx = np.where(limit_cross < 0)[0] + 1 # take next index
        threshold_cross = np.diff(np.sign(data - self.limit - self.threshold))
        threshold_cross_idx = np.where(threshold_cross > 0)[0] + 1 # take next index
        events = []
        if self._unfinished_event:
            events.append(self._unfinished_event)
            limit_cross_idx = np.r_[0, limit_cross_idx]
        evt_stop_idx = 0
        for limit_idx in limit_cross_idx:
            if limit_idx < evt_stop_idx:
                continue
            if limit_idx > 0: # Add new event
                events.append({"start_sidx": int(sidx[limit_idx]), "stop_sidx": None, "extrem_value": np.inf, "id": uuid.uuid4()})
            evt_stop_thr_idx = np.where(threshold_cross_idx > limit_idx)[0]
            if evt_stop_thr_idx.size > 0: # complete event
                logger.debug("Event Completed")
                evt_stop_idx = threshold_cross_idx[evt_stop_thr_idx[0]]
                events[-1]["stop_sidx"] = int(sidx[evt_stop_idx])
                events[-1]["extrem_value"] = min(events[-1]["extrem_value"], data[limit_idx:evt_stop_idx].min())
            else:
                events[-1]["extrem_value"] = min(events[-1]["extrem_value"], data[limit_idx:].min())
                break

        if events and events[-1]["stop_sidx"] is None:
            self._unfinished_event = events[-1]
        else:
            self._unfinished_event = {}
        
        return events

class EventDetectorLevelHigh(EventDetector):
    """Detects events where data exceeds a specified limit (LEVEL_HIGH).

    Inherits from EventDetector and overrides the _type to 'LEVEL_HIGH'.
    """
    def __init__(self, limit: float, threshold: float, observed_channel: DataChannelBuffer):
        """Initializes the Level High event detector.

        Args:
            limit: The upper threshold value.
            threshold: The buffer above the limit used to detect event completion.
            observed_channel: The channel being monitored for events.
        """
        super().__init__(limit, threshold, observed_channel)
        self._type = "LEVEL_HIGH"

    def process(self, start_sidx, stop_sidx):
        data, sidx = self._get_channel_data(start_sidx, stop_sidx)
        if data is None:
            return None
        limit_cross = np.diff(np.sign(data - self.limit))
        limit_cross_idx = np.where(limit_cross > 0)[0] + 1 # take next index
        threshold_cross = np.diff(np.sign(data - self.limit + self.threshold))
        threshold_cross_idx = np.where(threshold_cross < 0)[0] + 1 # take next index
        events = []
        if self._unfinished_event:
            events.append(self._unfinished_event)
            limit_cross_idx = np.r_[0, limit_cross_idx]
        evt_stop_idx = 0
        for limit_idx in limit_cross_idx:
            if limit_idx < evt_stop_idx:
                continue
            if limit_idx > 0: # Add new event
                events.append({"start_sidx": int(sidx[limit_idx]), "stop_sidx": None, "extrem_value": -np.inf, "id": uuid.uuid4()})
            evt_stop_thr_idx = np.where(threshold_cross_idx > limit_idx)[0]
            if evt_stop_thr_idx.size > 0: # complete event
                logger.debug("Event Completed")
                evt_stop_idx = threshold_cross_idx[evt_stop_thr_idx[0]]
                events[-1]["stop_sidx"] = int(sidx[evt_stop_idx])
                events[-1]["extrem_value"] = max(events[-1]["extrem_value"], data[limit_idx:evt_stop_idx].max())
            else:
                events[-1]["extrem_value"] = max(events[-1]["extrem_value"], data[limit_idx:].max())
                break

        if events and events[-1]["stop_sidx"] is None:
            self._unfinished_event = events[-1]
        else:
            self._unfinished_event = {}
        
        return events
        

class EventController(object):
    """Manages multiple event detectors and processes detected events over time.

    Attributes:
        PROCESSING_DELAY_SECONDS: Time (in seconds) after the latest sample to process events.
        _time_channel: The channel providing timestamp data.
        _sample_rate: Sampling rate of the system (samples per second).
        _event_detectors: List of registered event detectors.
        _last_processed_sidx: The last sample index processed.
        _unfinished_events: Dictionary mapping event IDs to unfinished Event objects.
    """
    PROCESSING_DELAY_SECONDS = 0.1

    def __init__(self, time_channel: AcqBuffer, sample_rate: float):
        """Initializes the EventController.

        Args:
            time_channel: The channel providing timestamp data.
            sample_rate: Sampling rate of the system (samples per second).
        """
        self._time_channel = time_channel
        self._sample_rate = sample_rate
        self._event_detectors: List[EventDetector] = []
        self._last_processed_sidx = 0
        self._unfinished_events: Dict[uuid.UUID: Event] = {}
    
    def add_event_detector(self, event_detector: EventDetector):
        self._event_detectors.append(event_detector)

    def process(self) -> List[Event]:
        start_acq_sidx = self._last_processed_sidx
        stop_acq_sidx = self._time_channel.sample_count - int(self.PROCESSING_DELAY_SECONDS*self._sample_rate)
        start_sidx_ts = int(self._time_channel.read_data_by_index(start_acq_sidx, start_acq_sidx+1)[0])
        self._last_processed_sidx = stop_acq_sidx
        if stop_acq_sidx <= 0:
            return []
        all_events = []
        for event_detector in self._event_detectors:
            events = event_detector.process(start_acq_sidx, stop_acq_sidx)
            if events:
                for event in events:
                    if event["id"] in self._unfinished_events:
                        start_ts = self._unfinished_events[event["id"]].start_ts
                    else:
                        start_ts = self._time_channel.read_data_by_index(event["start_sidx"], event["start_sidx"]+1)[0]/1e6
                    if event["stop_sidx"]:
                        stop_ts = self._time_channel.read_data_by_index(event["stop_sidx"], event["stop_sidx"]+1)[0]/1e6
                        # Delete finished event from map
                        if event["id"] in self._unfinished_events:
                            del self._unfinished_events[event["id"]]
                    else:
                        stop_ts = None
                    single_event = Event(start_ts=start_ts,
                                         stop_ts=stop_ts if stop_ts else None,
                                         start_sidx=event["start_sidx"],
                                         stop_sidx=event["stop_sidx"],
                                         extrem_value=event["extrem_value"],
                                         channel=event_detector.observed_channel.name,
                                         type=event_detector._type,
                                         id=event["id"])
                    all_events.append(single_event)
                    # Add unfinished event to map
                    if single_event.stop_sidx is None:
                        self._unfinished_events[single_event.id] = single_event
        return all_events


    
        


