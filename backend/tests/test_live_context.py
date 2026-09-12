import unittest

from app.live_context import _build_context


class LiveContextTests(unittest.TestCase):
    def test_context_preserves_provider_scope_and_24h_outlook(self):
        payload = {
            "current": {"time": "2026-09-12T12:00", "pm2_5": 42, "pm10": 70, "us_aqi": 110, "nitrogen_dioxide": 20},
            "hourly": {"time": [f"2026-09-12T{hour:02d}:00" for hour in range(24)], "pm2_5": list(range(24))},
        }
        result = _build_context(payload, "2026-09-12T06:30:00Z")
        self.assertEqual(len(result["outlook_24h"]), 24)
        self.assertEqual(result["source_kind"], "coarse_external_model_baseline")
        self.assertIn("not AeroShield", result["spatial_resolution_note"])


if __name__ == "__main__":
    unittest.main()
