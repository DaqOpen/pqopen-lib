import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class ZeroCrossDetector:
    """
    A class to detect zero-crossings in a signal after applying a low-pass filter.

    This detector processes input data to identify points where the signal crosses
    a specified threshold (both positive and negative), accounting for the phase delay
    introduced by the filter.

    Attributes:
        f_cutoff (float): Cutoff frequency of the low-pass filter in Hz.
        threshold (float): Threshold for detecting zero-crossings.
        samplerate (float): Sampling rate of the signal in Hz.
        _filter_coeff (tuple): Coefficients of the low-pass filter.
        _filter_zi (ndarray): Initial conditions for the filter.
        _last_filtered_sample (float): Last sample of the filtered data.
        _first_run (bool): Flag to indicate the first processing cycle.
        _last_zc_p (int or None): Last positive zero-crossing index from the previous block.
        _last_zc_n (int or None): Last negative zero-crossing index from the previous block.
        _filter_delay_samples (float): Filter delay in samples due to phase shift.
        _filtered_data (list): Temporary storage for filtered data during processing.
    """

    def __init__(self, f_cutoff: float, threshold: float, samplerate: float):
        """
        Initializes the ZeroCrossDetector with the specified parameters.

        Parameters:
            f_cutoff: Cutoff frequency of the low-pass filter in Hz.
            threshold: Threshold for detecting zero-crossings.
            samplerate: Sampling rate of the signal in Hz.
        """
        self.f_cutoff = f_cutoff
        self.threshold = threshold
        self.samplerate = samplerate

        # Design a Butterworth low-pass filter
        normal_cutoff = self.f_cutoff / (0.5 * self.samplerate)
        self._filter_coeff = signal.iirfilter(2, normal_cutoff, btype='lowpass', ftype='butter')
        self._filter_zi = np.zeros(len(self._filter_coeff[0]) - 1)

        self._last_filtered_sample = 0
        self._first_run = True
        self._last_zc_p = None
        self._last_zc_n = None
        self._last_zc_n_val = None

        # Calculate the filter delay in samples
        w, h = signal.freqz(self._filter_coeff[0], self._filter_coeff[1],
                            [2 * np.pi * self.f_cutoff / self.samplerate])
        self.filter_delay_samples = np.angle(h)[0] / (2 * np.pi) * self.samplerate / self.f_cutoff
        self.filtered_data = []

    def process(self, data: np.ndarray)-> list:
        """
        Processes a block of input data and detect zero-crossings.

        Parameters:
            data (ndarray): Input data block to be processed.

        Returns:
            list: A list of zero-crossing points (in sample indices).
        """
        if self._first_run:
            filtered_data = signal.lfilter(self._filter_coeff[0], self._filter_coeff[1], data)
            self._first_run = False
        else:
            filtered_data, _ = signal.lfilter(self._filter_coeff[0], self._filter_coeff[1], data, zi=self._filter_zi)

        #Update the filter state
        self._filter_zi = signal.lfiltic(
            self._filter_coeff[0], self._filter_coeff[1],
            filtered_data[-3:][::-1], data[-3:][::-1]
        )

        # Prepend the last sample from the previous block to maintain continuity
        filtered_data = np.r_[self._last_filtered_sample, filtered_data]

        # Calculate positive and negative crossings of threshold
        diff_data_p = np.diff(np.sign(filtered_data - self.threshold))
        diff_data_n = np.diff(np.sign(filtered_data + self.threshold))

        threshold_p_cross = np.where(diff_data_p > 0)[0] + 1
        threshold_n_cross = np.where(diff_data_n > 0)[0]

        self._last_filtered_sample = filtered_data[-1]

        # Include the last negative zero-crossing from the previous block
        if self._last_zc_n is not None:
            threshold_n_cross = np.r_[self._last_zc_n, threshold_n_cross]

        zero_crossings = []
        last_used_p_idx = -1

        # Put the data together to real rising zero crossing
        for n_idx in range(len(threshold_n_cross)):
            p_idx = threshold_p_cross.searchsorted(threshold_n_cross[n_idx], side='left')
            if p_idx >= len(threshold_p_cross):
                break
            if (n_idx + 1) < len(threshold_n_cross):
                # Skip n-threshold-crossings without a corresponding positive crossing
                if threshold_n_cross[n_idx + 1] <= threshold_p_cross[p_idx]:
                    continue
            x1 = threshold_n_cross[n_idx]
            x2 = threshold_p_cross[p_idx]
            y1 = filtered_data[x1] if x1 >= 0 else self._last_zc_n_val
            y2 = filtered_data[x2]
            k = (y2 - y1) / (x2 - x1)
            d = y1 - k * x1
            real_zc = -d/k
            #real_zc = (threshold_p_cross[p_idx] - threshold_n_cross[n_idx] + 1) / 2 + threshold_n_cross[n_idx]
            zero_crossings.append(real_zc + self.filter_delay_samples)
            last_used_p_idx = p_idx

        # Update the last negative threshold-crossing for the next block
        if last_used_p_idx < len(threshold_p_cross) and len(threshold_n_cross) > 0:
            self._last_zc_n = threshold_n_cross[-1] - len(data)
            if threshold_n_cross[-1] < 0:
                logger.debug("Zero Cross behind filtered data")
                self._last_zc_n_val = -self.threshold
            else:
                self._last_zc_n_val = filtered_data[threshold_n_cross[-1]]
        else:
            self._last_zc_n = None

        return zero_crossings
