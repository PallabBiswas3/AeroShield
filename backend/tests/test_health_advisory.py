import unittest

from app.health_advisory import generate_health_advisory


class HealthAdvisoryTests(unittest.TestCase):
    def test_upper_bound_drives_conservative_risk_level(self):
        result = generate_health_advisory(35, 125, "Ward 1")
        self.assertEqual(result["risk_level"], "SEVERE")
        self.assertIn("EN", result["languages"])
        self.assertIn("HI", result["languages"])

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_health_advisory(80, 60)


if __name__ == "__main__":
    unittest.main()
