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


def evaluate(train, calibration, test, residual=False, adaptive=False, gate=False):
    features = ['pm25', 'lag_1h', 'lag_3h', 'lag_24h', 'lag_168h', 'target_hour', 'target_weekday', 'target_month', 'location_id']
    # Fixed hyperparameters: no test-set selection. LightGBM handles missing lags.
    models = [LGBMRegressor(objective='quantile', alpha=q, n_estimators=200,
                           num_leaves=15, learning_rate=.05, random_state=42,
                           n_jobs=1, verbosity=-1) for q in [.05,.5,.95]]
    for model in models:
        model.fit(train[features], train.target_pm25-train.pm25 if residual else train.target_pm25)
    cal = np.sort(np.column_stack([m.predict(calibration[features]) for m in models]), axis=1)
    if residual:
        cal += calibration.pm25.to_numpy()[:,None]
    persistence_stations = []
    if gate:
        boundary = calibration.issue_time.sort_values().iloc[len(calibration)//2]
        selection = calibration.target_time < boundary
        for station in test.location_id.unique():
            mask = selection & calibration.location_id.eq(station)
            y = calibration.loc[mask, 'target_pm25'].to_numpy()
            if len(y) < 50 or np.mean((cal[mask,1]-y)**2) >= np.mean((calibration.loc[mask,'pm25'].to_numpy()-y)**2):
                persistence_stations.append(int(station))
        remaining = calibration.issue_time >= boundary
        calibration, cal = calibration.loc[remaining], cal[remaining]
        mask = calibration.location_id.isin(persistence_stations).to_numpy()
        cal[mask] = calibration.loc[mask,'pm25'].to_numpy()[:,None]
    scores = np.maximum(cal[:,0]-calibration.target_pm25.to_numpy(), calibration.target_pm25.to_numpy()-cal[:,2])
    rank = min(len(scores), int(np.ceil((len(scores)+1)*.9)))
    adjustment = max(0., float(np.sort(scores)[rank-1]))
    pred = np.sort(np.column_stack([m.predict(test[features]) for m in models]), axis=1)
    if residual:
        pred += test.pm25.to_numpy()[:,None]
    if gate:
        mask = test.location_id.isin(persistence_stations).to_numpy()
        pred[mask] = test.loc[mask,'pm25'].to_numpy()[:,None]
    result = test.copy()
    result['prediction'] = pred[:,1]
    result['lower'] = np.maximum(0., pred[:,0]-adjustment)
    result['upper'] = np.maximum(result.lower, pred[:,2]+adjustment)
    if adaptive:
        history = calibration[['location_id','target_time']].copy()
        history['score'] = scores
        observed = test[['location_id','target_time']].copy()
        observed['score'] = np.maximum(pred[:,0]-test.target_pm25.to_numpy(), test.target_pm25.to_numpy()-pred[:,2])
        history = pd.concat([history, observed]).sort_values('target_time')
        adjustments = []
        for row in test.itertuples():
            # Simulate feedback only after the target hour has completed.
            eligible = history[(history.target_time < row.issue_time) & (history.location_id == row.location_id)].tail(168)
            sample = eligible.score.to_numpy()
            if len(sample) < 50:
                adjustments.append(adjustment)
            else:
                rank = min(len(sample), int(np.ceil((len(sample)+1)*.9)))
                adjustments.append(max(0., float(np.sort(sample)[rank-1])))
        result['lower'] = np.maximum(0., pred[:,0]-np.array(adjustments))
        result['upper'] = np.maximum(result.lower, pred[:,2]+np.array(adjustments))
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
        'approach':'persistence_residual' if residual else 'absolute_target',
        'calibration':'delayed_station_168_scores' if adaptive else 'fixed_pooled',
        'persistence_gate_stations':persistence_stations,
        'rows':{'train':len(train),'calibration':len(calibration),'test':len(test)},
        'features':features, 'conformal_adjustment':adjustment,
        'pooled':report(result), 'stations':{str(k):report(v) for k,v in result.groupby('location_id')},
        'cams':'not evaluated: issue-time archive pending',
        'weather':'not included: forecast archive pending',
        'coverage_warning':'Temporal dependence and distribution shift invalidate an unconditional exchangeability guarantee.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rolling', action='store_true')
    parser.add_argument('--residual', action='store_true', help='Learn change from current PM2.5 rather than absolute concentration')
    parser.add_argument('--adaptive', action='store_true', help='Update station intervals using only matured target feedback')
    parser.add_argument('--gate', action='store_true', help='Select persistence on an earlier, separate selection window')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[2]/'data/raw_openaq/delhi_all_stations_2025.csv'
    hours, quality = station_hours(pd.read_csv(path))
    data = supervised(hours)
    splits = rolling_splits(data) if args.rolling else [chronological_split(data)]
    report = {'scope':'retrospective station-level history-only t+24 development benchmark',
              'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'quality':quality,
              'evaluation':'rolling origin' if args.rolling else 'single chronological holdout',
              'folds':[evaluate(*split, residual=args.residual, adaptive=args.adaptive, gate=args.gate) for split in splits]}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
