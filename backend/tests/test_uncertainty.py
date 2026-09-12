import unittest

import numpy as np

from app.ml.uncertainty import apply_conformal, calibrate_interval, interval_metrics


class UncertaintyTests(unittest.TestCase):
    def test_conformal_expands_undercovered_band(self):
        y = np.array([10.0, 20.0, 30.0, 40.0])
        lower = np.array([11.0, 21.0, 31.0, 41.0])
        upper = np.array([12.0, 22.0, 32.0, 42.0])
        qhat = calibrate_interval(y, lower, upper, coverage=0.75)
        lo, hi = apply_conformal(lower, upper, qhat)
        self.assertGreaterEqual(interval_metrics(y, lo, hi)["empirical_coverage"], 0.75)

    def test_interval_is_clipped_to_physical_range(self):
        lo, hi = apply_conformal(np.array([-5.0]), np.array([510.0]), 3.0)
        self.assertEqual(float(lo[0]), 0.0)
        self.assertEqual(float(hi[0]), 500.0)


if __name__ == "__main__":
    unittest.main()
