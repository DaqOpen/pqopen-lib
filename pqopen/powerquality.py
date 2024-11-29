# pqopen/powerquality.py

import numpy as np
from typing import Tuple

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