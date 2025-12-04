from datetime import datetime
import re
import numpy as np

def floor_timestamp(timestamp: float | int, interval_seconds: int, ts_resolution: str = "us"):
    """Floor eines Zeitstempels auf ein gegebenes Intervall."""
    if ts_resolution == "s":
        conversion_factor = 1.0
    elif ts_resolution == "ms":
        conversion_factor = 1_000.0
    elif ts_resolution == "us":
        conversion_factor = 1_000_000.0
    else:
        raise NotImplementedError(f"Time interval {ts_resolution} not implemented")
    if isinstance(timestamp, float):
        seconds = timestamp / conversion_factor
        floored_seconds = seconds - (seconds % interval_seconds)
        return floored_seconds * conversion_factor
    else:
        fraction = timestamp % int(conversion_factor*interval_seconds)
        return timestamp - fraction

class JsonDecimalLimiter(object):
    def __init__(self, decimal_places: int=2):
        """
        Limit float number precision to number of decimal places

        Parameters:
            json_string: the input string which should be converted
            decimal_places: number of remaining decimal places
        """
        self._decimal_places = decimal_places
        self._float_pattern = re.compile(r'(?<!")(-?\d+\.\d{'+str(decimal_places+1)+r',})(?!")')

    def process(self, json_string: str) -> str:
        """
        Limit the json_string's float objects decimal places

        Parameters:
            json_string (str): JSON String to be converted

        Returns:
            converted json_string
        """
        return self._float_pattern.sub(self._round_float_match, json_string)
    
    def _round_float_match(self, m):
        return format(round(float(m.group()), self._decimal_places), f'.{self._decimal_places}f')
    
def create_harm_corr_array(nom_frequency: float, num_harmonics: int, freq_response: tuple, interharm=False):
    if interharm:
        freqs = np.arange(0.5*nom_frequency, nom_frequency*(num_harmonics+1.5), nom_frequency)
    else:
        freqs = np.arange(0, nom_frequency*(num_harmonics+1), nom_frequency)
    freq_response_arr = np.array(freq_response)
    freq_corr = 1/np.interp(freqs, freq_response_arr[:,0], freq_response_arr[:,1])
    return freq_corr

def create_fft_corr_array(target_size: int, freq_nyq: float, freq_response: tuple):
    freqs = np.linspace(0, freq_nyq, target_size)
    freq_response_arr = np.array(freq_response)
    freq_corr = 1/np.interp(freqs, freq_response_arr[:,0], freq_response_arr[:,1])
    return freq_corr