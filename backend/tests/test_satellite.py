import unittest

from app.satellite import _parse_firms_csv, rank_fire_evidence


class SatelliteTests(unittest.TestCase):
    def test_firms_parser_rejects_points_outside_region(self):
        text = "latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight\n28.7,77.0,2026-09-12,0900,N,high,12.5,D\n10.0,10.0,2026-09-12,0900,N,high,50,D\n"
        fires = _parse_firms_csv(text)
        self.assertEqual(len(fires), 1)
        self.assertEqual(fires[0]["frp"], 12.5)

    def test_upwind_fire_scores_above_crosswind_fire(self):
        fires = [
            {"id": "upwind", "lat": 28.6, "lon": 77.0, "frp": 10},
            {"id": "crosswind", "lat": 28.8, "lon": 77.2, "frp": 10},
        ]
        ranked = rank_fire_evidence(28.6, 77.2, 270.0, fires)
        self.assertEqual(ranked[0]["id"], "upwind")
        self.assertGreater(ranked[0]["wind_compatibility"], ranked[1]["wind_compatibility"])


if __name__ == "__main__":
    unittest.main()
