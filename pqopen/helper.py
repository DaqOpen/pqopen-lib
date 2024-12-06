from datetime import datetime

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