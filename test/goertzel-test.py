import unittest
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pqopen.goertzel import calc_single_freq 

class TestSingleFrequency(unittest.TestCase):
    def test_fund_only(self):
        fs = 100
        t = np.arange(0, 1, 1/fs)
        u = np.sqrt(2)*np.sin(2*np.pi*t)
        amp, phase = calc_single_freq(u, 1, fs)

        self.assertAlmostEqual(1, amp)
        self.assertAlmostEqual(-90*np.pi/180, phase)

    def test_mixed(self):
        fs = 100
        t = np.arange(0, 1, 1/fs)
        u = np.sqrt(2)*np.sin(2*np.pi*t)
        u += 0.5*np.sqrt(2)*np.sin(2*3*np.pi*t)
        amp, phase = calc_single_freq(u, 1, fs)

        self.assertAlmostEqual(1, amp)
        self.assertAlmostEqual(-90*np.pi/180, phase)

if __name__ == "__main__":
    unittest.main()