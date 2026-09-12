import unittest

from app.ml.features import diurnal_traffic_factor, stability_class


class FeatureTests(unittest.TestCase):
    def test_weekday_rush_hour_exceeds_weekend(self):
        self.assertGreater(diurnal_traffic_factor(18, 0), diurnal_traffic_factor(18, 6))

    def test_stability_validates_hour(self):
        with self.assertRaises(ValueError):
            stability_class(3.0, 24)


if __name__ == "__main__":
    unittest.main()
