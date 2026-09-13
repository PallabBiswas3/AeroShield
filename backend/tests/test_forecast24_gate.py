import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from app.ml.forecast24 import supervised, chronological_split
from pipeline.benchmark_forecast24 import evaluate


class WrongResidual:
    def __init__(self, **kwargs):
        pass
    def fit(self, x, y):
        return self
    def predict(self, x):
        return np.full(len(x), 10.)


class SelectionTests(unittest.TestCase):
    def test_gate_uses_past_selection_labels_not_test_labels(self):
        hours = pd.DataFrame({'location_id':1, 'issue_time':pd.date_range('2025-01-01',periods=2000,freq='h',tz='UTC'), 'pm25':30.})
        train, cal, test = chronological_split(supervised(hours))
        with patch('pipeline.benchmark_forecast24.LGBMRegressor', WrongResidual):
            first = evaluate(train,cal,test,residual=True,adaptive=True,gate=True)
            changed = evaluate(train,cal,test.assign(target_pm25=300.),residual=True,adaptive=True,gate=True)
        self.assertEqual(first['persistence_gate_stations'],[1])
        self.assertEqual(first['persistence_gate_stations'],changed['persistence_gate_stations'])
        self.assertEqual(first['pooled']['model']['rmse'],0.)
