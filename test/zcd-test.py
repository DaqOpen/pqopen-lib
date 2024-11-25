import unittest
import sys
import os
import time
import numpy as np
from scipy.signal import chirp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pqopen.zcd import ZeroCrossDetector

class TestZeroCrossDetector(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by initializing the ZeroCrossDetector.
        """
        self.samplerate = 1000  # 5000 Hz sampling rate
        self.f_cutoff = 50  # 50 Hz cutoff frequency
        self.threshold = 0.1  # Threshold for zero-crossing detection
        self.detector = ZeroCrossDetector(f_cutoff=self.f_cutoff, threshold=self.threshold, samplerate=self.samplerate)

    def generate_sine_wave(self, freq: float, duration: float, amplitude: float = 1.0):
        """
        Generate a sine wave signal.

        Args:
            freq (float): Frequency of the sine wave in Hz.
            duration (float): Duration of the sine wave in seconds.
            amplitude (float): Amplitude of the sine wave.

        Returns:
            ndarray: Generated sine wave.
        """
        t = np.linspace(0, duration, int(self.samplerate * duration), endpoint=False)
        return amplitude * np.sin(2 * np.pi * freq * t)

    def test_zero_crossings_sine_wave_5Hz(self):
        """
        Test zero-crossing detection on a simple sine wave.
        """
        freq = 5  # 5 Hz sine wave
        duration = 1.1  # 1 second + 0.1 to respect filter delay
        sine_wave = self.generate_sine_wave(freq=freq, duration=duration)

        zero_crossings = self.detector.process(sine_wave)
        self.assertAlmostEqual(self.detector.filter_delay_samples, -5)

        # Expected zero-crossings: 5 Hz * 1 positive crossing per cycle * 1 second
        expected_crossings = np.arange(start=1, stop=6)*self.samplerate/freq
        self.assertEqual(len(zero_crossings), len(expected_crossings))
        self.assertIsNone(np.testing.assert_array_almost_equal(zero_crossings, expected_crossings, 0))

    def test_zero_crossings_sine_wave_50Hz(self):
        """
        Test zero-crossing detection on a simple sine wave.
        """
        freq = 50 
        duration = 1 
        sine_wave = self.generate_sine_wave(freq=freq, duration=duration)

        zero_crossings_1 = np.array(self.detector.process(sine_wave[:sine_wave.size//2]))
        zero_crossings_2 = np.array(self.detector.process(sine_wave[sine_wave.size//2:]))

        zero_crossings = np.concat((zero_crossings_1, zero_crossings_2 + sine_wave.size//2))

        # Expected zero-crossings: 50 Hz * 1 positive crossing per cycle * 1 second - 1
        expected_crossings = np.arange(start=1, stop=50)*self.samplerate/freq
        self.assertEqual(len(zero_crossings), len(expected_crossings))
        self.assertIsNone(np.testing.assert_array_almost_equal(zero_crossings, expected_crossings, 0))

    def test_zero_crossings_no_signal(self):
        """
        Test zero-crossing detection on a flat signal (should detect none).
        """
        data = np.zeros(1000)  # Flat signal
        zero_crossings = self.detector.process(data)
        self.assertEqual(len(zero_crossings), 0)

    def test_zero_crossings_chirp_signal(self):
        """
        Test zero-crossing detection on a chirp signal.
        """
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(self.samplerate * duration), endpoint=False)
        chirp_signal = chirp(t, f0=1, f1=50, t1=duration, method='linear')

        zero_crossings = self.detector.process(chirp_signal)

        # Test if zero crossings were detected (number will vary due to varying frequency)
        self.assertGreater(len(zero_crossings), 0)

    def test_zero_crossings_threshold_effect(self):
        """
        Test the effect of changing the threshold on zero-crossing detection.
        """
        freq = 5  # 5 Hz sine wave
        duration = 1  # 1 second
        amplitude = 0.5
        sine_wave = self.generate_sine_wave(freq=freq, duration=duration, amplitude=amplitude)

        # With a higher threshold, expect fewer zero-crossings
        self.detector.threshold = 1  # Adjust threshold to a high value
        zero_crossings_high_threshold = self.detector.process(sine_wave)

        # Reset detector with lower threshold
        self.detector = ZeroCrossDetector(f_cutoff=self.f_cutoff, threshold=self.threshold, samplerate=self.samplerate)
        zero_crossings_low_threshold = self.detector.process(sine_wave)

        self.assertGreater(len(zero_crossings_low_threshold), len(zero_crossings_high_threshold))

if __name__ == '__main__':
    unittest.main()
