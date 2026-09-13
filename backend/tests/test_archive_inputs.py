import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from pipeline.archive_inputs import forecast_at, capture


class ArchiveTests(unittest.TestCase):
    def test_provider_failure_creates_no_scenario_record(self):
        with TemporaryDirectory() as folder, patch('pipeline.archive_inputs.urlopen', side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError):
                capture('weather', 'https://example.test', folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_retrospective_download_not_prospective_snapshot(self):
        snapshot = self.snapshot()
        snapshot['provider'] = 'weather_history'
        with self.assertRaises(ValueError):
            forecast_at(snapshot, '2026-09-13T10:00:00Z', 'pm2_5')

    def snapshot(self):
        return {'provider': 'cams', 'available_at': '2026-09-13T09:59:00Z',
                'payload': {'utc_offset_seconds': 0, 'hourly': {
                    'time': ['2026-09-14T10:00'], 'pm2_5': [32.]}}}

    def test_exact_forecast(self):
        self.assertEqual(forecast_at(self.snapshot(), '2026-09-13T10:00:00Z', 'pm2_5'), 32.)

    def test_future_snapshot_rejected(self):
        with self.assertRaises(ValueError):
            forecast_at(self.snapshot(), '2026-09-13T09:00:00Z', 'pm2_5')

    def test_missing_target_not_interpolated(self):
        with self.assertRaises(ValueError):
            forecast_at(self.snapshot(), '2026-09-13T11:00:00Z', 'pm2_5')

    def test_missing_value_not_zero(self):
        snapshot = self.snapshot()
        snapshot['payload']['hourly']['pm2_5'] = [None]
        with self.assertRaises(ValueError):
            forecast_at(snapshot, '2026-09-13T10:00:00Z', 'pm2_5')

    def test_naive_issue_rejected(self):
        with self.assertRaises(ValueError):
            forecast_at(self.snapshot(), '2026-09-13T10:00:00', 'pm2_5')
