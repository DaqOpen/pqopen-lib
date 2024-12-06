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
import pqopen.powerquality as pq
from pqopen.helper import floor_timestamp

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
                 zcd_cutoff_freq: float = 50.0,
                 zcd_threshold: float = 1.0,
                 zcd_minimum_freq: float = 10,
                 nominal_frequency: float = 50.0,
                 nper: int = 10):
        """
        Initializes a PowerSystem object.

        Parameters:
            zcd_channel: Channel buffer for zero-crossing detection.
            input_samplerate: Sampling rate of the input signal.
            zcd_cutoff_freq: Cutoff frequency for zero-crossing detection. Defaults to 50.0.
            zcd_threshold: Threshold for zero-crossing detection. Defaults to 1.0.
            zcd_minimum_freq: Minimum frequency for valid zero crossings. Defaults to 10.
            nominal_frequency: Nominal system frequency. Defaults to 50.0.
            nper: Number of periods for calculations. Defaults to 10.
        """
        
        self._zcd_channel = zcd_channel
        self._samplerate = input_samplerate
        self._zcd_cutoff_freq = zcd_cutoff_freq
        self._zcd_threshold = zcd_threshold
        self._zcd_minimum_frequency = zcd_minimum_freq
        self.nominal_frequency = nominal_frequency
        self.nper = nper
        self._phases: List[PowerPhase] = []
        self._features = {"harmonics": 0, 
                          "fluctuation": False,
                          "nper_abs_time_sync": False}
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

    def enable_fluctuation_calculation(self, nominal_voltage: float = 230, pst_interval_sec: int = 600):
        self._features["fluctuation"] = True
        self._nominal_voltage = nominal_voltage
        self._pst_interval_sec = pst_interval_sec
        self._pst_next_round_ts = 0
        self._pst_last_calc_sidx = 0
        self._update_calc_channels()

    def enable_nper_abs_time_sync(self, time_channel: AcqBuffer, interval_sec: int = 600):
        """
        Enables synchronisation of multi-period calculation to absolute rounded timestamp.
        Complies to IEC 61000-4-30 overlapping

        Parameters:
            time_channel: Channel buffer for time information
            interval_sec: Resync interval in seconds
        """
        self._features["nper_abs_time_sync"] = True
        self._time_channel = time_channel
        self._resync_interval_sec = interval_sec
        self._next_round_ts = 0

    def _resync_nper_abs_time(self, zc_idx: int):
        if not self._features["nper_abs_time_sync"]:
            return None
        last_zc_ts = int(self._time_channel.read_data_by_index(self._zero_crossings[zc_idx], self._zero_crossings[zc_idx]+1)[0])
        if self._next_round_ts == 0:
            self._next_round_ts = int(floor_timestamp(last_zc_ts, self._resync_interval_sec, ts_resolution="us")+self._resync_interval_sec*1_000_000)
        if last_zc_ts > self._next_round_ts:
            logger.debug("Passed rounded timestamp - resync")
            last_nper_ts = self._time_channel.read_data_by_index(self._zero_crossings[zc_idx-self.nper], self._zero_crossings[zc_idx])
            next_round_sidx = self._zero_crossings[zc_idx-self.nper] + np.searchsorted(last_nper_ts, self._next_round_ts)
            # Forward Zero-Cross counter to comply to overlap according to IEC 61000-4-30
            back_idx = -1
            self._zero_cross_counter -= 1 # Rewind one zc back
            while self._zero_crossings[back_idx+1+zc_idx] > next_round_sidx:
                self._zero_cross_counter += 1 # Forward zc count
                back_idx -= 1
            logger.debug(f"Rewind index: {back_idx:d}, {self._zero_crossings[zc_idx]:d}, {self._zero_crossings[back_idx]:d}, next_round_sample_idx: {next_round_sidx:d}")
            self._next_round_ts = floor_timestamp(last_zc_ts, self._resync_interval_sec, ts_resolution="us"+self._resync_interval_sec*1_000_000)
        
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
            if len(self._phases) == 3:
                self._calc_channels["multi_period"]["voltage"]["unbal_0"] = DataChannelBuffer('U_unbal_0', agg_type='mean', unit="%")
                self._calc_channels["multi_period"]["voltage"]["unbal_2"] = DataChannelBuffer('U_unbal_2', agg_type='mean', unit="%")
            if "current" in phase._calc_channels[agg_interval]:
                self._calc_channels["one_period"]["power"]["p_avg"] = DataChannelBuffer('P_1p', agg_type='mean', unit="W")
                self._calc_channels["multi_period"]["power"]["p_avg"] = DataChannelBuffer('P', agg_type='mean', unit="W")

            for agg_interval, phys_types in self._calc_channels.items():
                for phys_type, calc_type in phys_types.items():
                    tmp = {channel.name: channel for channel in calc_type.values()}
                self.output_channels.update(tmp)

            if self._features["fluctuation"]:
                for phase in self._phases:
                    phase._voltage_fluctuation_processor = pq.VoltageFluctuation(samplerate=self._samplerate,
                                                                                 nominal_volt=self._nominal_voltage,
                                                                                 nominal_freq=self.nominal_frequency)
    
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
            if self._zero_cross_counter <= 1:
                continue
            # Process one period calculation, start with second zc
            self._process_one_period(self._zero_crossings[-2], self._zero_crossings[-1])
            if ((self._zero_cross_counter-1) % self.nper) == 0 and (self._zero_cross_counter > self.nper):
                # Process multi-period
                self._process_multi_period(self._zero_crossings[-self.nper - 1], self._zero_crossings[-1])
                self._resync_nper_abs_time(-1)
                self._process_fluctuation_calc(self._zero_crossings[-self.nper - 1], self._zero_crossings[-1])
        
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
            # Read phase angle of phases's voltage if available
            if "fund_phi" in phase._calc_channels["multi_period"]["voltage"]:
                u_phi = phase._calc_channels["multi_period"]["voltage"]["fund_phi"].last_sample_value
            else:
                u_phi = 0.0
            if u_phi < 0:
                u_phi += 360
            u_phi_samples = int(self._samplerate/frequency*u_phi/360)
            phase_period_start_sidx = period_start_sidx - u_phi_samples
            phase_period_stop_sidx = period_stop_sidx - u_phi_samples
            phase_period_half_sidx = phase_period_start_sidx + int((phase_period_stop_sidx - phase_period_start_sidx)/2)
            u_values = phase._u_channel.read_data_by_index(phase_period_start_sidx, phase_period_stop_sidx)
            for phys_type, output_channel in phase._calc_channels["one_period"]["voltage"].items():
                if phys_type == "trms":
                    u_rms = np.sqrt(np.mean(np.power(u_values, 2)))
                    output_channel.put_data_single(phase_period_stop_sidx, u_rms)
            for phys_type, output_channel in phase._calc_channels["half_period"]["voltage"].items():
                if phys_type == "trms":
                    # First half period
                    u_rms = np.sqrt(np.mean(np.power(u_values[:len(u_values)//2], 2)))
                    output_channel.put_data_single(phase_period_half_sidx, u_rms)
                    # Second half period
                    u_rms = np.sqrt(np.mean(np.power(u_values[len(u_values)//2:], 2)))
                    output_channel.put_data_single(phase_period_stop_sidx, u_rms)

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

    def _process_multi_period(self, start_sidx: int, stop_sidx: int):
        """
        Processes data for multi periods, calculating rms, harmonics

        Parameters:
            start_sidx: Start sample index of the interval.
            stop_sidx: Stop sample index of the interval.
        """
        phi_ref = 0.0
        for phase in self._phases:
            u_values = phase._u_channel.read_data_by_index(start_sidx, stop_sidx)
            if self._features["harmonics"]:
                data_fft_U = pq.resample_and_fft(u_values)
                u_h_mag, u_h_phi = pq.calc_harmonics(data_fft_U, self.nper, self._features["harmonics"])
                u_ih_mag = pq.calc_interharmonics(data_fft_U, self.nper, self._features["harmonics"])
            for phys_type, output_channel in phase._calc_channels["multi_period"]["voltage"].items():
                if phys_type == "trms":
                    u_rms = np.sqrt(np.mean(np.power(u_values, 2)))
                    output_channel.put_data_single(stop_sidx, u_rms)
                if phys_type == "fund_rms":
                    output_channel.put_data_single(stop_sidx, u_h_mag[1])
                if phys_type == "fund_phi":
                    if phase._number == 1:
                        phi_ref = u_h_phi[1]
                    output_channel.put_data_single(stop_sidx, pq.normalize_phi(u_h_phi[1]-phi_ref))
                if phys_type == "harm_rms":
                    output_channel.put_data_single(stop_sidx, u_h_mag)
                if phys_type == "iharm_rms":
                    output_channel.put_data_single(stop_sidx, u_ih_mag)
                if phys_type == "thd":
                    output_channel.put_data_single(stop_sidx, pq.calc_thd(u_h_mag))

            if phase._i_channel:
                i_values = phase._i_channel.read_data_by_index(start_sidx, stop_sidx)
                if self._features["harmonics"]:
                    data_fft_I = pq.resample_and_fft(i_values)
                    i_h_mag, i_h_phi = pq.calc_harmonics(data_fft_I, self.nper, self._features["harmonics"])
                    i_ih_mag = pq.calc_interharmonics(data_fft_I, self.nper, self._features["harmonics"])
                for phys_type, output_channel in phase._calc_channels["multi_period"]["current"].items():
                    if phys_type == "trms":
                        i_rms = np.sqrt(np.mean(np.power(i_values, 2)))
                        output_channel.put_data_single(stop_sidx, i_rms)
                    if phys_type == "fund_rms":
                        output_channel.put_data_single(stop_sidx, i_h_mag[1])
                    if phys_type == "fund_phi":
                        output_channel.put_data_single(stop_sidx, i_h_phi[1] - phi_ref)
                    if phys_type == "harm_rms":
                        output_channel.put_data_single(stop_sidx, i_h_mag)
                    if phys_type == "iharm_rms":
                        output_channel.put_data_single(stop_sidx, i_ih_mag)
                    if phys_type == "thd":
                        output_channel.put_data_single(stop_sidx, pq.calc_thd(i_h_mag))
                    
                for phys_type, output_channel in phase._calc_channels["multi_period"]["power"].items():
                    if phys_type == "p_avg":
                        p_avg = np.mean(u_values * i_values)
                        output_channel.put_data_single(stop_sidx, p_avg)
    
    def _process_fluctuation_calc(self, start_sidx: int, stop_sidx: int):
        if not self._features["fluctuation"]:
            return None
        for phase in self._phases:
            u_raw = phase._u_channel.read_data_by_index(start_sidx, stop_sidx)
            u_hp_rms, _ = phase._calc_channels["half_period"]["voltage"]["trms"].read_data_by_acq_sidx(start_sidx, stop_sidx)
            phase._voltage_fluctuation_processor.process(start_sidx, u_hp_rms, u_raw)
        stop_ts = int(self._time_channel.read_data_by_index(stop_sidx, stop_sidx+1)[0])
        if self._pst_next_round_ts == 0:
            self._pst_next_round_ts = floor_timestamp(stop_ts, self._pst_interval_sec, ts_resolution="us")+self._pst_interval_sec*1_000_000
        # Calculate Pst and forward next timestamps due to interval
        if (stop_ts > self._pst_next_round_ts):
            logger.debug("Calculating Pst")
            ts_between_start_stop = self._time_channel.read_data_by_index(start_sidx, stop_sidx)
            calc_stop_sidx = start_sidx + np.searchsorted(ts_between_start_stop, self._pst_next_round_ts)
            if self._pst_last_calc_sidx == 0:
                self._pst_last_calc_sidx = calc_stop_sidx
                self._pst_next_round_ts = self._pst_next_round_ts + self._pst_interval_sec*1_000_000
                return None
            for phase in self._phases:
                pst = phase._voltage_fluctuation_processor.calc_pst(self._pst_last_calc_sidx, calc_stop_sidx)
                phase._calc_channels["multi_period"]["voltage"]["pst"].put_data_single(calc_stop_sidx, pst)
            self._pst_last_calc_sidx = calc_stop_sidx
            self._pst_next_round_ts = floor_timestamp(stop_ts, self._pst_interval_sec, ts_resolution="us")+self._pst_interval_sec*1_000_000

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
        self._voltage_fluctuation_processor: pq.VoltageFluctuation = None

    def update_calc_channels(self, features):
        self._calc_channels = {"half_period": {}, "one_period": {}, "one_period_ovlp": {}, "multi_period": {}}
        # Create Voltage Channels
        self._calc_channels["half_period"]["voltage"] = {}
        self._calc_channels["one_period"]["voltage"] = {}
        self._calc_channels["multi_period"]["voltage"] = {}
        self._calc_channels["one_period"]["voltage"]["trms"] = DataChannelBuffer('U{:s}_1p_rms'.format(self.name), agg_type='rms', unit="V")
        self._calc_channels["multi_period"]["voltage"]["trms"] = DataChannelBuffer('U{:s}_rms'.format(self.name), agg_type='rms', unit="V")

        if "harmonics" in features and features["harmonics"]:
            self._calc_channels["multi_period"]["voltage"]["fund_rms"] = DataChannelBuffer('U{:s}_H1_rms'.format(self.name), agg_type='rms', unit="V")
            self._calc_channels["multi_period"]["voltage"]["fund_phi"] = DataChannelBuffer('U{:s}_H1_phi'.format(self.name), agg_type='phi', unit="°")
            self._calc_channels["multi_period"]["voltage"]["harm_rms"] = DataChannelBuffer('U{:s}_H_rms'.format(self.name), sample_dimension=features["harmonics"]+1, agg_type='rms', unit="V")
            self._calc_channels["multi_period"]["voltage"]["iharm_rms"] = DataChannelBuffer('U{:s}_IH_rms'.format(self.name), sample_dimension=features["harmonics"]+1, agg_type='rms', unit="V")
            self._calc_channels["multi_period"]["voltage"]["thd"] = DataChannelBuffer('U{:s}_THD'.format(self.name), unit="%")

        if "fluctuation" in features and features["fluctuation"]:
            self._calc_channels["half_period"]["voltage"]["trms"] = DataChannelBuffer('U{:s}_hp_rms'.format(self.name), agg_type='rms', unit="V")
            self._calc_channels["multi_period"]["voltage"]["pst"] = DataChannelBuffer('U{:s}_pst'.format(self.name), agg_type='max', unit="")

        # Create Current Channels
        if self._i_channel:
            self._calc_channels["one_period"]["current"] = {}
            self._calc_channels["multi_period"]["current"] = {}
            self._calc_channels["one_period"]["current"]["trms"] = DataChannelBuffer('I{:s}_1p_rms'.format(self.name), agg_type='rms', unit="A")
            self._calc_channels["multi_period"]["current"]["trms"] = DataChannelBuffer('I{:s}_rms'.format(self.name), agg_type='rms', unit="A")

            if "harmonics" in features and features["harmonics"]:
                self._calc_channels["multi_period"]["current"]["fund_rms"] = DataChannelBuffer('I{:s}_H1_rms'.format(self.name), agg_type='rms', unit="A")
                self._calc_channels["multi_period"]["current"]["fund_phi"] = DataChannelBuffer('I{:s}_H1_phi'.format(self.name), agg_type='phi', unit="°")
                self._calc_channels["multi_period"]["current"]["harm_rms"] = DataChannelBuffer('I{:s}_H_rms'.format(self.name), sample_dimension=features["harmonics"]+1, agg_type='rms', unit="A")
                self._calc_channels["multi_period"]["current"]["iharm_rms"] = DataChannelBuffer('I{:s}_IH_rms'.format(self.name), sample_dimension=features["harmonics"]+1, agg_type='rms', unit="A")
                self._calc_channels["multi_period"]["current"]["thd"] = DataChannelBuffer('I{:s}_THD'.format(self.name), unit="%")

            # Create Power Channels
            self._calc_channels["one_period"]["power"] = {}
            self._calc_channels["multi_period"]["power"] = {}
            self._calc_channels["one_period"]["power"]['p_avg'] = DataChannelBuffer('P{:s}_1p'.format(self.name), agg_type='mean', unit="W")
            self._calc_channels["one_period"]["power"]['q_avg'] = DataChannelBuffer('Q{:s}_1p'.format(self.name), agg_type='mean', unit="var")
            self._calc_channels["multi_period"]["power"]['p_avg'] = DataChannelBuffer('P{:s}'.format(self.name), agg_type='mean', unit="W")
            self._calc_channels["multi_period"]["power"]['q_avg'] = DataChannelBuffer('Q{:s}'.format(self.name), agg_type='mean', unit="var")
            self._calc_channels["multi_period"]["power"]['p_fund_mag'] = DataChannelBuffer('P{:s}_H1'.format(self.name), agg_type='mean', unit="W")
            self._calc_channels["multi_period"]["power"]['q_fund_mag'] = DataChannelBuffer('Q{:s}_H1'.format(self.name), agg_type='mean', unit="var")
