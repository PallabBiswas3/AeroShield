"""Run: python -m pipeline.benchmark_forecast24 (prints reproducible JSON)."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from app.ml.forecast24 import station_hours, supervised, chronological_split, rolling_splits


def metrics(y, prediction):
    error = np.asarray(y)-np.asarray(prediction)
    return {'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.mean(abs(error)))}


def evaluate(train, calibration, test):
    features = ['pm25', 'lag_1h', 'lag_3h', 'lag_24h', 'lag_168h', 'target_hour', 'target_weekday', 'target_month', 'location_id']
    # Fixed hyperparameters: no test-set selection. LightGBM handles missing lags.
    models = [LGBMRegressor(objective='quantile', alpha=q, n_estimators=200,
                           num_leaves=15, learning_rate=.05, random_state=42,
                           n_jobs=1, verbosity=-1) for q in [.05,.5,.95]]
    for model in models:
        model.fit(train[features], train.target_pm25)
    cal = np.sort(np.column_stack([m.predict(calibration[features]) for m in models]), axis=1)
    scores = np.maximum(cal[:,0]-calibration.target_pm25.to_numpy(), calibration.target_pm25.to_numpy()-cal[:,2])
    rank = min(len(scores), int(np.ceil((len(scores)+1)*.9)))
    adjustment = max(0., float(np.sort(scores)[rank-1]))
    pred = np.sort(np.column_stack([m.predict(test[features]) for m in models]), axis=1)
    result = test.copy()
    result['prediction'] = pred[:,1]
    result['lower'] = np.maximum(0., pred[:,0]-adjustment)
    result['upper'] = np.maximum(result.lower, pred[:,2]+adjustment)
    def report(frame):
        model = metrics(frame.target_pm25, frame.prediction)
        baseline = metrics(frame.target_pm25, frame.pm25)
        matched = frame.dropna(subset=['weekly_persistence'])
        return {'n':len(frame), 'model':model, 'persistence':baseline,
                'rmse_skill':1-model['rmse']/baseline['rmse'] if baseline['rmse'] else None,
                'coverage':float(((frame.target_pm25>=frame.lower)&(frame.target_pm25<=frame.upper)).mean()),
                'mean_width':float((frame.upper-frame.lower).mean()),
                'weekly_matched':{'n':len(matched), 'model':metrics(matched.target_pm25, matched.prediction),
                                  'weekly':metrics(matched.target_pm25, matched.weekly_persistence)} if len(matched) else None}
    return {
        'test_start':str(test.issue_time.min()), 'test_end':str(test.issue_time.max()),
        'rows':{'train':len(train),'calibration':len(calibration),'test':len(test)},
        'features':features, 'conformal_adjustment':adjustment,
        'pooled':report(result), 'stations':{str(k):report(v) for k,v in result.groupby('location_id')},
        'cams':'not evaluated: issue-time archive pending',
        'weather':'not included: forecast archive pending',
        'coverage_warning':'Temporal dependence and distribution shift invalidate an unconditional exchangeability guarantee.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rolling', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[2]/'data/raw_openaq/delhi_all_stations_2025.csv'
    hours, quality = station_hours(pd.read_csv(path))
    data = supervised(hours)
    splits = rolling_splits(data) if args.rolling else [chronological_split(data)]
    report = {'scope':'retrospective station-level history-only t+24 development benchmark',
              'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'quality':quality,
              'evaluation':'rolling origin' if args.rolling else 'single chronological holdout',
              'folds':[evaluate(*split) for split in splits]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
