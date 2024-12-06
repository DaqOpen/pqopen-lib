# pqopen/powerquality.py

import numpy as np
from scipy import signal
from typing import Tuple
from daqopen.channelbuffer import DataChannelBuffer

def calc_harmonics(fft_data: np.ndarray, num_periods: int=10, num_harmonics: int=100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate harmonic rms and phase for a given FFT dataset.
    Grouping according to IEC 61000-4-7

    Parameters:
        fft_data: The FFT data array.
        num_periods: The number of fundamental periods in the origin time domain data
        num_harmonics: The number of harmonics to calculate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            harm_rms (np.ndarray): The RMS values of the harmonics.
            harm_phi (np.ndarray): The phase angles of the harmonics in degrees.
    """
    harm_rms_middle = np.abs(fft_data[0:num_harmonics*num_periods+1:num_periods])
    harm_rms_left = np.r_[0, np.abs(fft_data[num_periods-1:num_harmonics*num_periods+1:num_periods])]
    harm_rms_right = np.abs(fft_data[1:num_harmonics*num_periods+1+1:num_periods])
    harm_rms = np.sqrt(np.power(harm_rms_left, 2)+ np.power(harm_rms_middle, 2)+np.power(harm_rms_right, 2))
    harm_phi = np.angle(fft_data[0:num_harmonics*num_periods+1:num_periods], deg=True)
    return harm_rms, harm_phi

def calc_interharmonics(fft_data: np.ndarray, num_periods: int=10, num_iharmonics: int=100) -> np.ndarray:
    """
    Calculate interharmonic rms for a given FFT dataset.
    Grouping according to IEC 61000-4-7

    Parameters:
        fft_data: The FFT data array.
        num_periods: The number of fundamental periods in the origin time domain data
        num_iharmonics: The number of interharmonics to calculate.

    Returns:
        iharm_rms (np.ndarray): The RMS values of the interharmonics.
    """
    reshaped_data = np.abs(fft_data[:num_periods*(num_iharmonics+1)]).reshape((-1, num_periods))
    iharm_rms = np.sqrt(np.sum(np.power(reshaped_data[:,2:9], 2), axis=1))
    return iharm_rms

def calc_thd(harm_rms: np.ndarray, max_harmonic: int = 40, min_harmonic: int = 2, fund_rms: float = None) -> float:
    """
    Calculate the Total Harmonic Distortion (THD).

    Parameters:
        harm_rms: The RMS values of the harmonics.
        max_harmonic: The maximum harmonic order to include in the THD calculation.
        min_harmonic: The minimum harmonic order to include in the THD calculation.
        fund_rms: The RMS value of the fundamental frequency. If None, the first harmonic is used.

    Returns:
        float: The calculated THD percentage.
    """
    if fund_rms is None:
        thd = np.sqrt(np.sum(np.power(harm_rms[min_harmonic:max_harmonic+1]/harm_rms[1], 2)))*100
    else:
        thd = np.sqrt(np.sum(np.power(harm_rms[min_harmonic:max_harmonic+1]/fund_rms, 2)))*100
    return thd

def resample_and_fft(data: np.ndarray, resample_size: int = None) -> np.ndarray:
    """
    Resample the input data to a specified size and compute its FFT.

    Parameters:
        data: The input data array.
        resample_size: The size to which the data should be resampled. 
            If None, the size is determined as the next power of 2.

    Returns:
        np.ndarray: The FFT of the resampled data, scaled by sqrt(2) and normalized by the resample size.
    """
    if not resample_size:
        resample_size = 2**int(np.ceil(np.log2(data.size)))
    x_data = np.arange(len(data))
    x = np.linspace(0, len(data), resample_size, endpoint=False)
    data_resampled = np.interp(x, x_data, data)
    return np.fft.rfft(data_resampled)/resample_size*np.sqrt(2)

def normalize_phi(phi: float) -> float:
    """Normalize phase angle to +-180°

    Parameters:
        phi: Phase angle to normalize

    Returns:
        normalized phase angle
    """
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi

class VoltageFluctuation(object):
    OUTPUT_SAMPLERATE_DECIMATION = 5
    WANTED_DECIMATED_SAMPLERATE = 5000
    CALIBRATION_FACTOR = 309602.0

    def __init__(self, samplerate: float, nominal_volt: float = 230, nominal_freq: float = 50):
        self._samplerate = samplerate
        self._calc_samplerate_decimation = max(1, int(self._samplerate / self.WANTED_DECIMATED_SAMPLERATE))
        self._decimated_samplerate = self._samplerate/self._calc_samplerate_decimation
        self._nominal_freq = nominal_freq
        self._nominal_volt = nominal_volt
        self._init_filter()
        self._pinst_channel = DataChannelBuffer("Pinst", size=int(1_000*self._decimated_samplerate/self.OUTPUT_SAMPLERATE_DECIMATION))
        self.steady_state = False
        self.processed_samples = 0
        self._perc_names = [0.1, 0.7, 1, 1.5, 2.2, 3, 4, 6, 8, 10, 13, 17, 30, 50, 80]
        self._percentiles = (100 - np.array(self._perc_names))
        self._normalized_signal = []
        self._next_reduction_start_idx = 0
        
    def _init_filter(self):
        stage0_tp_norm_cutoff = 500 / (0.5 * self._samplerate)
        self.stage0_tp_filter_coeff = signal.iirfilter(2, stage0_tp_norm_cutoff, btype='lowpass', ftype='butter')
        self.stage0_tp_filter_zi = np.zeros(len(self.stage0_tp_filter_coeff[0])-1)
        stage1_tp_norm_cutoff = 1/(27.3*2*np.pi) / (0.5 * self._nominal_freq*2)
        self.stage1_tp_filter_coeff = signal.iirfilter(1, stage1_tp_norm_cutoff, btype='lowpass', ftype='butter')
        self.stage1_tp_filter_zi = np.ones(len(self.stage1_tp_filter_coeff[0])-1)*self._nominal_volt
        stage3_hp_norm_cutoff = 0.05 / (0.5 * self._decimated_samplerate)
        self.stage3_hp_filter_coeff = signal.iirfilter(1, stage3_hp_norm_cutoff, btype='highpass', ftype='butter')
        self.stage3_hp_filter_zi = np.zeros(len(self.stage3_hp_filter_coeff[0])-1)
        stage3_tp_norm_cutoff = 35 / (0.5 * self._decimated_samplerate)
        self.stage3_tp_filter_coeff = signal.iirfilter(6, stage3_tp_norm_cutoff, btype='lowpass', ftype='butter')
        self.stage3_tp_filter_zi = np.zeros(len(self.stage3_tp_filter_coeff[0])-1)
        k = 1.74802
        lam = 2*np.pi*4.05981
        w1 = 2*np.pi*9.15494
        w2 = 2*np.pi*2.27979
        w3 = 2*np.pi*1.22535
        w4 = 2*np.pi*21.9
        num1 = [0, k * w1, 0]
        den1 = [1, 2 * lam, w1**2]
        num2 = [0, 1 / w2, 1]
        den2 = [1 / (w3 * w4), 1 / w3 + 1 / w4, 1];
        self.stage3_weight_filter_coeff = signal.bilinear(np.convolve(num1, num2), signal.convolve(den1, den2), self._decimated_samplerate); 
        self.stage3_weight_filter_zi = np.zeros(len(self.stage3_weight_filter_coeff[0])-1)
        stage4_tp_norm_cutoff = 1/(0.3*2*np.pi) / (0.5 * self._decimated_samplerate)
        self.stage4_tp_filter_coeff = signal.iirfilter(1, stage4_tp_norm_cutoff, btype='lowpass', ftype='butter')
        self.stage4_tp_filter_zi = np.zeros(len(self.stage4_tp_filter_coeff[0])-1)
    
    def process(self, start_sidx: int, hp_data: np.ndarray, raw_data: np.ndarray):
        """
        """
        stage0_tp_filtered_data,_ = signal.lfilter(self.stage0_tp_filter_coeff[0], self.stage0_tp_filter_coeff[1], raw_data, zi=self.stage0_tp_filter_zi)
        self.stage0_tp_filter_zi = signal.lfiltic(self.stage0_tp_filter_coeff[0], 
                                                  self.stage0_tp_filter_coeff[1], 
                                                  stage0_tp_filtered_data[-3:][::-1],
                                                  raw_data[-3:][::-1])
        
        samples_skip_next_start = len(stage0_tp_filtered_data) % self._calc_samplerate_decimation
        stage0_tp_filtered_data = stage0_tp_filtered_data[self._next_reduction_start_idx::self._calc_samplerate_decimation]
        if samples_skip_next_start:
            self._next_reduction_start_idx += self._calc_samplerate_decimation - samples_skip_next_start
            if self._next_reduction_start_idx >= self._calc_samplerate_decimation:
                self._next_reduction_start_idx %= self._calc_samplerate_decimation
        stage1_tp_filtered_data,_ = signal.lfilter(self.stage1_tp_filter_coeff[0], self.stage1_tp_filter_coeff[1], hp_data, zi=self.stage1_tp_filter_zi)
        self.stage1_tp_filter_zi = signal.lfiltic(self.stage1_tp_filter_coeff[0], 
                                                  self.stage1_tp_filter_coeff[1], 
                                                  stage1_tp_filtered_data[-2:][::-1],
                                                  hp_data[-2:][::-1])
        blk_size = int(len(stage0_tp_filtered_data)/len(hp_data))
        for idx,val in enumerate(stage1_tp_filtered_data[:-1]):
            stage0_tp_filtered_data[idx*blk_size:(idx+1)*blk_size] /= val
        stage0_tp_filtered_data[(idx+1)*blk_size:] /= stage1_tp_filtered_data[-1]
        stage2_output = np.power(stage0_tp_filtered_data, 2)
        stage3_hp_filtered_data,_ = signal.lfilter(self.stage3_hp_filter_coeff[0], self.stage3_hp_filter_coeff[1], stage2_output, zi=self.stage3_hp_filter_zi)
        self.stage3_hp_filter_zi = signal.lfiltic(self.stage3_hp_filter_coeff[0], 
                                                  self.stage3_hp_filter_coeff[1], 
                                                  stage3_hp_filtered_data[-2:][::-1],
                                                  stage2_output[-2:][::-1])
        stage3_tp_filtered_data,_ = signal.lfilter(self.stage3_tp_filter_coeff[0], self.stage3_tp_filter_coeff[1], stage3_hp_filtered_data, zi=self.stage3_tp_filter_zi)
        self.stage3_tp_filter_zi = signal.lfiltic(self.stage3_tp_filter_coeff[0], 
                                                  self.stage3_tp_filter_coeff[1], 
                                                  stage3_tp_filtered_data[-7:][::-1],
                                                  stage3_hp_filtered_data[-7:][::-1])
        stage3_weight_filtered_data,_ = signal.lfilter(self.stage3_weight_filter_coeff[0], self.stage3_weight_filter_coeff[1], stage3_tp_filtered_data, zi=self.stage3_weight_filter_zi)
        self.stage3_weight_filter_zi = signal.lfiltic(self.stage3_weight_filter_coeff[0], 
                                                  self.stage3_weight_filter_coeff[1], 
                                                  stage3_weight_filtered_data[-len(self.stage3_weight_filter_coeff[1])-1:][::-1],
                                                  stage3_tp_filtered_data[-len(self.stage3_weight_filter_coeff[1])-1:][::-1])
        stage3_output = np.power(stage3_weight_filtered_data, 2)
        stage4_tp_filtered_data,_ = signal.lfilter(self.stage4_tp_filter_coeff[0], self.stage4_tp_filter_coeff[1], stage3_output, zi=self.stage4_tp_filter_zi)
        self.stage4_tp_filter_zi = signal.lfiltic(self.stage4_tp_filter_coeff[0], 
                                                  self.stage4_tp_filter_coeff[1], 
                                                  stage4_tp_filtered_data[-2:][::-1],
                                                  stage3_output[-2:][::-1])
        # Append buffer since steady state
        self.processed_samples += len(raw_data)
        # Steady State after approx. 20 Seconds
        if self.processed_samples / self._samplerate > 20:
            self.steady_state = True
        if self.steady_state:
            data_to_buffer = (stage4_tp_filtered_data*self.CALIBRATION_FACTOR)[::self.OUTPUT_SAMPLERATE_DECIMATION]
            try:
                output_sidx = np.linspace(start=start_sidx, 
                                          stop=start_sidx + raw_data.size, 
                                          num=len(data_to_buffer),
                                          endpoint=False)
                self._pinst_channel.put_data_multi(output_sidx, data_to_buffer)
            except:
                pass
        
    def calc_pst(self, start_sidx: int, stop_sidx: int):
        if not self.steady_state:
            return None
        perc = np.percentile(self._pinst_channel.read_data_by_acq_sidx(start_sidx, stop_sidx)[0], self._percentiles)
        P = {str(self._perc_names[idx]): perc[idx] for idx in range(len(perc))}
        P50s = (P['30'] + P['50'] + P['80'])/3
        P10s = (P['6'] + P['8'] + P['10'] + P['13'] + P['17'])/5
        P3s = (P['2.2'] + P['3'] + P['4'])/3
        P1s = (P['0.7'] + P['1'] + P['1.5'])/3
        Pst = np.sqrt(0.0314*P['0.1'] + 0.0525*P1s + 0.0657*P3s + 0.28*P10s + 0.08*P50s)
        return Pst