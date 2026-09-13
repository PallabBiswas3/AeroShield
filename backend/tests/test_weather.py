import unittest
from unittest.mock import patch, Mock
from datetime import datetime
import requests
from app.ml import weather


class WeatherTests(unittest.TestCase):
    def test_cached_weather_keeps_original_retrieval_time(self):
        response = Mock()
        response.json.return_value = {'hourly': {'time': ['2026-09-13T12:00'],
            'wind_speed_10m': [3], 'wind_direction_10m': [120]}}
        with patch.object(weather, '_CACHE', {}), patch.object(weather, '_FAILURES', {}), patch.object(weather.requests, 'get', return_value=response) as get:
            first = weather.get_forecast_weather(28.6, 77.2, datetime(2026, 9, 13, 12))
            second = weather.get_forecast_weather(28.6, 77.2, datetime(2026, 9, 13, 12))
            self.assertEqual(first['fetched_at'], second['fetched_at'])
            self.assertEqual(get.call_count, 1)

    def test_provider_failure_is_throttled(self):
        with patch.object(weather, '_CACHE', {}), patch.object(weather, '_FAILURES', {}), patch.object(weather.requests, 'get', side_effect=requests.HTTPError()) as get:
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    weather.get_forecast_weather(28.6,77.2,datetime(2026,9,13,12))
            self.assertEqual(get.call_count,1)

    def test_partial_wind_never_claims_complete_forecast(self):
        response=Mock()
        response.json.return_value={'hourly':{'time':['2026-09-13T12:00'], 'wind_speed_10m':[None], 'wind_direction_10m':[120]}}
        with patch.object(weather, '_CACHE', {}), patch.object(weather, '_FAILURES', {}), patch.object(weather.requests,'get',return_value=response):
            with self.assertRaises(RuntimeError):
                weather.get_forecast_weather(28.6,77.2,datetime(2026,9,13,12))
