# pqopen/powersystem.py

"""
Module for creating power system objects.

This module provides classes and methods to define power systems, including phases, 
zero-crossing detection, and calculation of electrical quantities like voltage, current,
and power.

Classes:
    PowerSystem: Represents the overall power system, allowing configuration, data processing, and analysis.
    PowerPhase: Represents a single phase of the power system.

Imports:
    - numpy: For numerical calculations.
    - List: For type hinting lists.
    - logging: For logging messages.
    - AcqBuffer, DataChannelBuffer: From daqopen.channelbuffer for data handling.
    - ZeroCrossDetector: From pqopen.zcd for detecting zero crossings in signals.
"""

import numpy as np
from typing import List
import logging

from daqopen.channelbuffer import AcqBuffer, DataChannelBuffer
from pqopen.zcd import ZeroCrossDetector

logger = logging.getLogger(__name__)

class PowerSystem(object):
    """
    Represents the overall power system, including zero-crossing detection,
    phase management, and power system calculations.

    Attributes:
        _zcd_channel (AcqBuffer): Channel buffer for zero-crossing detection data.
        _samplerate (float): Sampling rate of the input signal.
        _time_channel (AcqBuffer): Optional time channel buffer.
        _zcd_cutoff_freq (float): Cutoff frequency for the zero-crossing detector.
        _zcd_threshold (float): Threshold for zero-crossing detection.
        _zcd_minimum_frequency (float): Minimum frequency for valid zero-crossing detection.
        nominal_frequency (float): Nominal frequency of the power system.
        nper (Optional[int]): Number of periods for analysis (if applicable).
        _phases (List[PowerPhase]): List of phases in the power system.
        _features (dict): Configuration for harmonic and fluctuation calculations.
        output_channels (dict): Dictionary of output data channels.
    """
    def __init__(self, 
                 zcd_channel: AcqBuffer, 
                 input_samplerate: float,
                 time_channel: AcqBuffer = None,
                 zcd_cutoff_freq: float = 50.0,
                 zcd_threshold: float = 1.0,
                 zcd_minimum_freq: float = 10,
                 nominal_frequency: float = 50.0,
                 nper = None):
        """
        Initializes a PowerSystem object.

        Parameters:
            zcd_channel: Channel buffer for zero-crossing detection.
            input_samplerate: Sampling rate of the input signal.
            time_channel: Optional time channel buffer.
            zcd_cutoff_freq: Cutoff frequency for zero-crossing detection. Defaults to 50.0.
            zcd_threshold: Threshold for zero-crossing detection. Defaults to 1.0.
            zcd_minimum_freq: Minimum frequency for valid zero crossings. Defaults to 10.
            nominal_frequency: Nominal system frequency. Defaults to 50.0.
            nper: Number of periods for calculations. Defaults to None.
        """
        
        self._zcd_channel = zcd_channel
        self._samplerate = input_samplerate
        self._time_channel = time_channel
        self._zcd_cutoff_freq = zcd_cutoff_freq
        self._zcd_threshold = zcd_threshold
        self._zcd_minimum_frequency = zcd_minimum_freq
        self.nominal_frequency = nominal_frequency
        self.nper = nper
        self._phases: List[PowerPhase] = []
        self._features = {"harmonics": 0, 
                          "fluctuation": False}
        self._prepare_calc_channels()
        self.output_channels = {}
        self._last_processed_sidx = 0
        self._zero_cross_detector = ZeroCrossDetector(f_cutoff=self._zcd_cutoff_freq,
                                                      threshold=self._zcd_threshold,
                                                      samplerate=self._samplerate)
        self._zero_crossings = [0]*20
        self._zero_cross_counter = 0
        self._calculation_mode = "NORMAL"
        self._last_known_freq = self.nominal_frequency


    def _prepare_calc_channels(self):
        self._calc_channels = {"half_period":      {"voltage": {}, "current": {}, "power": {}}, 
                                "one_period":       {"voltage": {}, "current": {}, "power": {}}, 
                                "one_period_ovlp":  {"voltage": {}, "current": {}, "power": {}}, 
                                "multi_period":     {"voltage": {}, "current": {}, "power": {}}}

    def add_phase(self, u_channel: AcqBuffer, i_channel: AcqBuffer = None, name: str = ""):
        """
        Adds a phase to the power system.

        Parameters:
            u_channel: Voltage channel buffer.
            i_channel: Current channel buffer. Defaults to None.
            name: Name of the phase. Defaults to an empty string.
        """
        if not name:
            name = str(len(self._phases)+1)
        self._phases.append(PowerPhase(u_channel=u_channel, i_channel=i_channel, number=len(self._phases)+1, name=name))
        self._update_calc_channels()

    def enable_harmonic_calculation(self, num_harmonics: int = 50):
        """
        Enables harmonic analysis for the power system.

        Parameters:
            num_harmonics: Number of harmonics to calculate. Defaults to 50.
        """
        self._features["harmonics"] = num_harmonics
        self._update_calc_channels()

    def enable_fluctuation_calculation(self):
        self._features["fluctuation"] = True
        self._update_calc_channels()

    def _update_calc_channels(self):
        self.output_channels = {}
        for phase in self._phases:
            phase.update_calc_channels(self._features)
            for agg_interval, phys_types in phase._calc_channels.items():
                for phys_type, calc_type in phys_types.items():
                    tmp = {channel.name: channel for channel in calc_type.values()}
                    self.output_channels.update(tmp)
        self._calc_channels["one_period"]["power"]["freq"] = DataChannelBuffer('Freq', agg_type='mean', unit="Hz")
        if self._phases:
            for agg_interval in phase._calc_channels:
                self._calc_channels[agg_interval]["voltage"]["rms"] = DataChannelBuffer('U_1p_rms', agg_type='rms', unit="V")
                if len(self._phases) == 3:
                    self._calc_channels[agg_interval]["voltage"]["unbal_0"] = DataChannelBuffer('U_unbal_0', agg_type='mean', unit="%")
                    self._calc_channels[agg_interval]["voltage"]["unbal_2"] = DataChannelBuffer('U_unbal_2', agg_type='mean', unit="%")
                if "current" in phase._calc_channels[agg_interval]:
                    self._calc_channels[agg_interval]["current"]["rms"] = DataChannelBuffer('I_1p_rms', agg_type='rms', unit="V")
                    self._calc_channels[agg_interval]["power"]["p_avg"] = DataChannelBuffer('P', agg_type='mean', unit="W")

            for agg_interval, phys_types in self._calc_channels.items():
                for phys_type, calc_type in phys_types.items():
                    tmp = {channel.name: channel for channel in calc_type.values()}
                self.output_channels.update(tmp)
    
    def process(self):
        """
        Processes new data samples, performing zero-crossing detection and calculations for each period.
        """
        # Process new samples in buffer
        if not self._phases:
            raise ValueError("No phases defined yet")
        start_acq_sidx = self._last_processed_sidx
        stop_acq_sidx = self._phases[0]._u_channel.sample_count

        zero_crossings = self._detect_zero_crossings(start_acq_sidx, stop_acq_sidx)
        for zc in zero_crossings:
            self._zero_cross_counter += 1
            actual_zc = int(np.round(zc)) + start_acq_sidx
            # Ignore Zero Crossing before actual stop Sample IDX
            if actual_zc >= stop_acq_sidx:
                logger.warning("Warning: Detected Zerocross before actual sample count")
                continue
            self._zero_crossings.pop(0)
            self._zero_crossings.append(actual_zc)
            # Process one period calculation
            self._process_one_period(self._zero_crossings[-2], self._zero_crossings[-1])
        
        self._last_processed_sidx = stop_acq_sidx

    def _process_one_period(self, period_start_sidx: int, period_stop_sidx: int):
        """
        Processes data for a single period, calculating voltage, current, and power.

        Parameters:
            period_start_sidx: Start sample index of the period.
            period_stop_sidx: Stop sample index of the period.
        """
        frequency = self._samplerate/(period_stop_sidx - period_start_sidx)
        self._calc_channels["one_period"]['power']['freq'].put_data_single(period_stop_sidx, frequency)
        for phase in self._phases:
            u_values = phase._u_channel.read_data_by_index(period_start_sidx, period_stop_sidx)
            for phys_type, output_channel in phase._calc_channels["one_period"]["voltage"].items():
                if phys_type == "trms":
                    u_rms = np.sqrt(np.mean(np.power(u_values, 2)))
                    output_channel.put_data_single(period_stop_sidx, u_rms)

            if phase._i_channel:
                i_values = phase._i_channel.read_data_by_index(period_start_sidx, period_stop_sidx)
                for phys_type, output_channel in phase._calc_channels["one_period"]["current"].items():
                    if phys_type == "trms":
                        i_rms = np.sqrt(np.mean(np.power(i_values, 2)))
                        output_channel.put_data_single(period_stop_sidx, i_rms)
                for phys_type, output_channel in phase._calc_channels["one_period"]["power"].items():
                    if phys_type == "p_avg":
                        p_avg = np.mean(u_values * i_values)
                        output_channel.put_data_single(period_stop_sidx, p_avg)
                

    def _detect_zero_crossings(self, start_acq_sidx: int, stop_acq_sidx: int) -> List[int]:
        """
        Detects zero crossings in the signal.

        Parameters:
            start_acq_sidx: Start sample index for detection.
            stop_acq_sidx: Stop sample index for detection.

        Returns:
            List[int]: Detected zero-crossing indices.
        """
        zcd_data = self._zcd_channel.read_data_by_index(start_idx=start_acq_sidx, stop_idx=stop_acq_sidx)
        zero_crossings = self._zero_cross_detector.process(zcd_data)
        if not zero_crossings:
            if (self._zero_crossings[-1] + self._samplerate/self._zcd_minimum_frequency - self._zero_cross_detector.filter_delay_samples) < stop_acq_sidx:
                zero_crossings.append(self._zero_crossings[-1] + self._samplerate/self._last_known_freq - start_acq_sidx)
                while (zero_crossings[-1] + self._samplerate/self._zcd_minimum_frequency - self._zero_cross_detector.filter_delay_samples) < (stop_acq_sidx - start_acq_sidx):
                    additional_zc = zero_crossings[-1] + self._samplerate/self._last_known_freq
                    if additional_zc < stop_acq_sidx - self._zero_cross_detector.filter_delay_samples:
                        zero_crossings.append(additional_zc)
                        logger.debug(f"Added virtual zero crossing: idx={additional_zc:f}")
                if self._calculation_mode == "NORMAL":
                    freq_last_1s,ts = self._calc_channels["one_period"]['power']['freq'].read_data_by_acq_sidx(self._zero_crossings[-1] - self._samplerate, self._zero_crossings[-1])
                    if len(freq_last_1s) > 0:
                        self._last_known_freq = freq_last_1s.mean()
                        if self._last_known_freq < self._zcd_minimum_frequency:
                            self._last_known_freq = self.nominal_frequency
                self._calculation_mode = "FALLBACK"
        else:
            self._calculation_mode = "NORMAL"
        
        # Remove Non monotonic rising zero crossings
        filtered_zero_crossings = []
        for idx,zc in enumerate(zero_crossings):
            if (int(zc)+start_acq_sidx) >= self._zero_crossings[-1] or np.isnan(self._zero_crossings[-1]):
                filtered_zero_crossings.append(zc)
        return filtered_zero_crossings

    def get_aggregated_data(self, start_acq_sidx: int, stop_acq_sidx: int) -> dict:
        """
        Retrieves aggregated data for the specified sample range.

        Parameters:
            start_acq_sidx: Start sample index.
            stop_acq_sidx: Stop sample index.

        Returns:
            dict: Aggregated data values.
        """
        output_values = {}
        for ch_name, channel in self.output_channels.items():
            ch_data = channel.read_agg_data_by_acq_sidx(start_acq_sidx, stop_acq_sidx)
            output_values[ch_name] = ch_data
        return output_values  

    
class PowerPhase(object):
    """
    Represents a single phase in the power system.

    Attributes:
        _u_channel (AcqBuffer): Voltage channel buffer for the phase.
        _i_channel (AcqBuffer): Current channel buffer for the phase.
        _number (int): Identifier number for the phase.
        name (str): Name of the phase.
        _calc_channels (dict): Dictionary for storing calculated data channels.
    """
    def __init__(self, u_channel: AcqBuffer, i_channel: AcqBuffer = None, number: int = 1, name: str = ""):
        """
        Initializes a PowerPhase object.

        Args:
            u_channel: Voltage channel buffer.
            i_channel: Current channel buffer. Defaults to None.
            number: Phase number. Defaults to 1.
            name: Name of the phase. Defaults to an empty string.
        """
        self._u_channel = u_channel
        self._i_channel = i_channel
        self._number = number
        self.name = name
        self._calc_channels = {}

    def update_calc_channels(self, features):
        self._calc_channels = {"half_period": {}, "one_period": {}, "one_period_ovlp": {}, "multi_period": {}}
        # Create Voltage Channels
        self._calc_channels["one_period"]["voltage"] = {}
        self._calc_channels["one_period"]["voltage"]["trms"] = DataChannelBuffer('U{:s}_1p_rms'.format(self.name), agg_type='rms', unit="V")
        self._calc_channels["one_period"]["voltage"]["fund_rms"] = DataChannelBuffer('U{:s}_H1_rms'.format(self.name), agg_type='rms', unit="V")

        if "harmonics" in features and features["harmonics"]:
            self._calc_channels["multi_period"]["voltage"] = {}
            self._calc_channels["multi_period"]["voltage"]["harm_rms"] = DataChannelBuffer('U{:s}_H_rms'.format(self.name), sample_dimension=features["harmonics"], agg_type='rms', unit="V")

        # Create Current Channels
        if self._i_channel:
            self._calc_channels["one_period"]["current"] = {}
            self._calc_channels["one_period"]["current"]["trms"] = DataChannelBuffer('I{:s}_1p_rms'.format(self.name), agg_type='rms', unit="A")
            self._calc_channels["one_period"]["current"]["fund_rms"] = DataChannelBuffer('I{:s}_1p_H1_rms'.format(self.name), agg_type='rms', unit="A")

            if "harmonics" in features and features["harmonics"]:
                self._calc_channels["multi_period"]["current"] = {}
                self._calc_channels["multi_period"]["current"]["harm_rms"] = DataChannelBuffer('I{:s}_H_rms'.format(self.name), sample_dimension=features["harmonics"], agg_type='rms', unit="A")

            # Create Power Channels
            self._calc_channels["one_period"]["power"] = {}
            self._calc_channels["one_period"]["power"]['p_avg'] = DataChannelBuffer('P{:s}_1p'.format(self.name), agg_type='mean', unit="W")
            self._calc_channels["one_period"]["power"]['p_fund_mag'] = DataChannelBuffer('P{:s}_1p_H1'.format(self.name), agg_type='mean', unit="W")
            self._calc_channels["one_period"]["power"]['q_avg'] = DataChannelBuffer('Q{:s}_1p'.format(self.name), agg_type='mean', unit="var")
            self._calc_channels["one_period"]["power"]['q_fund_mag'] = DataChannelBuffer('Q{:s}_1p_H1'.format(self.name), agg_type='mean', unit="var")
