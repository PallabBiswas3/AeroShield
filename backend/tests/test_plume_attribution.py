import unittest

from app.ml.plume_math import calculate_source_attribution_ensemble


SOURCES = [
    {"source_id": 1, "name": "Upwind works", "type": "Industrial Stack", "lat": 28.60, "lon": 77.00, "intensity": 800},
    {"source_id": 2, "name": "Crosswind works", "type": "Industrial Stack", "lat": 28.75, "lon": 77.20, "intensity": 800},
]


class PlumeAttributionTests(unittest.TestCase):
    def test_upwind_source_is_ranked_first(self):
        result = calculate_source_attribution_ensemble(
            hotspot_lat=28.60, hotspot_lon=77.20, wind_speed=4.0,
            wind_dir_deg=270.0, sources_list=SOURCES, hour=12,
            simulations=200, seed=7,
        )
        self.assertEqual(result[0]["source_id"], 1)
        self.assertGreater(result[0]["rank_one_probability"], 70)
        self.assertAlmostEqual(sum(item["attribution_probability"] for item in result), 100.0, delta=0.2)

    def test_seed_makes_attribution_reproducible(self):
        args = dict(hotspot_lat=28.60, hotspot_lon=77.20, wind_speed=4.0, wind_dir_deg=270.0, sources_list=SOURCES, hour=12, simulations=100, seed=11)
        self.assertEqual(calculate_source_attribution_ensemble(**args), calculate_source_attribution_ensemble(**args))

    def test_reports_inventory_conditional_semantics(self):
        result = calculate_source_attribution_ensemble(
            hotspot_lat=28.60, hotspot_lon=77.20, wind_speed=4.0,
            wind_dir_deg=270.0, sources_list=SOURCES, hour=12,
            simulations=100, seed=3,
        )
        self.assertIn("attribution_share", result[0])
        self.assertIn("conditional", result[0]["interpretation"].lower())
        self.assertGreater(result[0]["active_simulation_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
