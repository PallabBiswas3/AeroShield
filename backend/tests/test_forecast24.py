import unittest
import pandas as pd
from app.ml.forecast24 import supervised, chronological_split, available_forecasts, rolling_splits


class Forecast24Tests(unittest.TestCase):
    def test_rolling_windows_are_disjoint_and_purged(self):
        data = supervised(self.hours())
        folds = list(rolling_splits(data, calibration_days=3, test_days=4, minimum_train_days=10))
        self.assertGreater(len(folds), 1)
        previous = None
        for train, cal, test in folds:
            self.assertLess(train.target_time.max(), cal.issue_time.min())
            self.assertLess(cal.target_time.max(), test.issue_time.min())
            if previous is not None:
                self.assertLess(previous, test.issue_time.min())
            previous = test.issue_time.max()

    def hours(self):
        return pd.DataFrame({'location_id': 1, 'issue_time': pd.date_range('2025-01-01', periods=1000, freq='h', tz='UTC'), 'pm25': range(1000)})

    def test_target_exactly_24_hours_and_no_row_shift_across_gap(self):
        hours = self.hours().drop(index=24)
        data = supervised(hours)
        self.assertFalse(data.issue_time.eq(hours.issue_time.iloc[0]).any())
        row = data.iloc[0]
        self.assertEqual(row.target_pm25 - row.pm25, 24)
        self.assertEqual(row.target_time - row.issue_time, pd.Timedelta(hours=24))

    def test_labels_do_not_cross_partition_boundaries(self):
        train, cal, test = chronological_split(supervised(self.hours()))
        self.assertLess(train.target_time.max(), cal.issue_time.min())
        self.assertLess(cal.target_time.max(), test.issue_time.min())

    def test_future_available_weather_rejected(self):
        frame = pd.DataFrame({'issue_time':['2025-01-01'], 'available_at':['2025-01-02'], 'valid_time':['2025-01-02']})
        with self.assertRaises(ValueError):
            available_forecasts(frame)

    def test_stations_never_share_lags(self):
        a = self.hours()
        b = a.assign(location_id=2, pm25=a.pm25+10000)
        result = supervised(pd.concat([a,b]))
        self.assertTrue((result.target_pm25-result.pm25).eq(24).all())
