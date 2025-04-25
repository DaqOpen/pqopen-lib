from datetime import datetime
import re

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