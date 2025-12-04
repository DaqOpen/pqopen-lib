import numpy as np
try:
    from numba import njit, float32, float64
except ImportError:
    # Dummy-Decorator if numba not available
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    float32 = float64 = None

@njit(cache=True)
def calc_single_freq(x: np.array, f_hz: float, fs: float) -> tuple[float, float]:
    """Compute the amplitude and phase of a specific frequency component
    in a signal using the Goertzel algorithm.

    Parameters:
        x: Input signal (time-domain samples).
        f_hz: Target frequency in Hertz (Hz).
        fs: Sampling frequency of the signal in Hertz (Hz).

    Returns:
        tuple:
            amp: Amplitude of the frequency component.
            phase: Phase of the frequency component in radians.
    """
    N = len(x)                  # Number of samples
    k = (f_hz * N) / fs         # Corresponding DFT bin index (can be fractional)
    w = 2 * np.pi * k / N       # Angular frequency for the bin
    cw = np.cos(w)              # Cosine component
    c = 2 * cw                  # Multiplier used in recurrence relation
    sw = np.sin(w)              # Sine component
    z1, z2 = 0, 0               # Initialize state variables

    # Recursive filter loop
    for n in range(N):
        z0 = x[n] + c * z1 - z2  # Apply recurrence relation
        z2 = z1                  # Shift states
        z1 = z0

    # Compute real and imaginary parts of the result
    ip = cw * z1 - z2   # In-phase (real) component
    qp = sw * z1        # Quadrature (imaginary) component

    # Compute amplitude and phase of the frequency component
    amp = np.sqrt((ip**2 + qp**2)/2) / (N / 2)
    phase = np.arctan2(qp, ip)

    return amp, phase

def calc_rms_trapz(values: np.array, start_frac: float, end_frac: float, frequency: float, samplerate: float):
    u2 = values * values
    s = 0.0
    s += 0.5 * (u2[0] + u2[1]) * start_frac # left edge
    s += 0.5*u2[1:-2].sum() + 0.5*u2[2:-1].sum() # middle
    s += 0.5 * (u2[-2] + u2[-1]) * end_frac # right edge
    return (s * frequency / samplerate) ** 0.5

@njit(fastmath=True)
def fast_interp(y, out_len):
    n = y.size
    scale = (n - 1) / out_len
    out = np.empty(out_len, dtype=y.dtype)
    for i in range(out_len):
        x = i * scale
        j = int(x)
        t = x - j
        if j+1 < n:
            out[i] = (1-t)*y[j] + t*y[j+1]
        else:
            out[i] = y[-1]
    return out

# >>>>>>> Pre-Compile for float32 und float64 <<<<<<<
if float32 is not None and float64 is not None:
    sigs = [
        "(float32[:], float32, float32)",
        "(float64[:], float64, float64)"
    ]
    for sig in sigs:
        calc_single_freq.compile(sig)