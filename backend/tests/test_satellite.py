import unittest
from unittest.mock import Mock, patch

import requests

import app.satellite as satellite
from app.satellite import _parse_firms_csv, _safe_failure_reason, rank_fire_evidence


class SatelliteTests(unittest.TestCase):
    def test_firms_response_is_reused_until_ttl_expires(self):
        response = Mock()
        response.text = "latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight\n28.7,77.0,2026-09-12,0900,N,high,12.5,D\n"
        now = [100]
        with patch.object(satellite, "_memory_cache", None), \
             patch.object(satellite, "monotonic", side_effect=lambda: now[0]), \
             patch.object(satellite, "_write_cache"), \
             patch.object(satellite.requests, "get", return_value=response) as get, \
             patch.dict(satellite.os.environ, {"NASA_FIRMS_MAP_KEY": "test-key"}):
            first = satellite.get_satellite_fires()
            now[0] = 101
            second = satellite.get_satellite_fires()
            self.assertEqual(first["mode"], "live")
            self.assertEqual(second["mode"], "memory_cache")
            self.assertEqual(first["fetched_at"], second["fetched_at"])
            self.assertFalse(second["stale"])
            self.assertEqual(get.call_count, 1)

            now[0] = 401
            self.assertEqual(satellite.get_satellite_fires()["mode"], "live")
            self.assertEqual(get.call_count, 2)

    def test_key_change_invalidates_memory_cache(self):
        response = Mock(text="latitude,longitude\n")
        with patch.object(satellite, "_memory_cache", None), \
             patch.object(satellite, "_write_cache"), \
             patch.object(satellite.requests, "get", return_value=response) as get, \
             patch.dict(satellite.os.environ, {"NASA_FIRMS_MAP_KEY": "first-key"}):
            satellite.get_satellite_fires()
            satellite.os.environ["NASA_FIRMS_MAP_KEY"] = "second-key"
            satellite.get_satellite_fires()
            self.assertEqual(get.call_count, 2)

    def test_failed_firms_request_is_retried_after_short_ttl(self):
        now = [100]
        with patch.object(satellite, "_memory_cache", None), \
             patch.object(satellite, "monotonic", side_effect=lambda: now[0]), \
             patch.object(satellite, "_read_cache", return_value=None), \
             patch.object(satellite.requests, "get", side_effect=requests.Timeout()) as get, \
             patch.dict(satellite.os.environ, {"NASA_FIRMS_MAP_KEY": "test-key"}):
            self.assertEqual(satellite.get_satellite_fires()["mode"], "unavailable")
            now[0] = 101
            self.assertEqual(satellite.get_satellite_fires()["mode"], "unavailable")
            self.assertEqual(get.call_count, 1)
            now[0] = 161
            satellite.get_satellite_fires()
            self.assertEqual(get.call_count, 2)

    def test_http_failure_does_not_expose_key_bearing_url(self):
        response = requests.Response()
        response.status_code = 403
        response.url = "https://firms.example/api/SECRET-MAP-KEY/area"
        error = requests.HTTPError("403 for key URL", response=response)
        reason = _safe_failure_reason(error)
        self.assertEqual(reason, "NASA FIRMS returned HTTP 403")
        self.assertNotIn("SECRET-MAP-KEY", reason)

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
