import unittest

import numpy as np
import pandas as pd

from pipeline.step3_etl_and_train import engineer_features


class PipelineFeatureTests(unittest.TestCase):
    def test_lag_names_represent_exact_elapsed_hours(self):
        times = pd.to_datetime([
            "2025-01-01T00:00:00Z",
            "2025-01-01T02:00:00Z",
            "2025-01-01T03:00:00Z",
        ])
        frame = pd.DataFrame({
            "location_id": [1, 1, 1], "datetime_hour": times,
            "latitude": [28.628] * 3, "longitude": [77.209] * 3,
            "pm25": [10.0, 20.0, 30.0], "wind_speed_met": [3.0] * 3,
            "wind_dir_met": [270.0] * 3, "blh_met": [500.0] * 3,
            "road_density_m": [1000.0] * 3,
        })
        result = engineer_features(frame, [])
        self.assertTrue(np.isnan(result.iloc[1]["lag_1h_pm25"]))
        self.assertEqual(result.iloc[2]["lag_1h_pm25"], 20.0)

    def test_training_coordinates_use_fixed_serving_origin(self):
        frame = pd.DataFrame({
            "location_id": [1], "datetime_hour": pd.to_datetime(["2025-01-01T00:00:00Z"]),
            "latitude": [28.628], "longitude": [77.209], "pm25": [10.0],
            "wind_speed_met": [3.0], "wind_dir_met": [270.0], "blh_met": [500.0],
            "road_density_m": [1000.0],
        })
        result = engineer_features(frame, [])
        self.assertEqual(result.iloc[0]["x"], 0.0)
        self.assertEqual(result.iloc[0]["y"], 0.0)


if __name__ == "__main__":
    unittest.main()
