"""Day-ahead benchmark foundations. No imputation of targets or missing hours."""
import numpy as np
import pandas as pd


def station_hours(raw):
    """Aggregate raw PM2.5 into completed UTC hours, labelled by hour END.

    Historical files lack ingestion timestamps. Availability at hour end is an
    optimistic retrospective assumption, not a claim of operational availability.
    """
    df = raw.loc[raw.parameter.eq('pm25')].copy()
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    df['time'] = pd.to_datetime(df.datetimeUtc, utc=True, errors='coerce')
    units = df.unit.astype(str).str.replace('μ', 'µ').str.lower()
    valid = df.time.notna() & np.isfinite(df.value) & df.value.ge(0) & units.isin(['µg/m³', 'µg/m3', 'ug/m3'])
    quality = {'pm25_rows': len(df), 'rejected_rows': int((~valid).sum()),
               'availability_assumption': 'hour-end; historical ingestion time unknown'}
    df = df.loc[valid].drop_duplicates(['location_id', 'time', 'value'])
    df['issue_time'] = df.time.dt.floor('h') + pd.Timedelta(hours=1)
    hourly = df.groupby(['location_id', 'issue_time']).value.agg(pm25='mean', sample_count='count').reset_index()
    quality['station_hours'] = len(hourly)
    return hourly, quality


def supervised(hours):
    df = hours.copy()
    for lag in [1, 3, 24, 168]:
        past = hours[['location_id', 'issue_time', 'pm25']].copy()
        past.issue_time += pd.Timedelta(hours=lag)
        df = df.merge(past.rename(columns={'pm25': f'lag_{lag}h'}), on=['location_id', 'issue_time'], how='left', validate='one_to_one')
    target = hours[['location_id', 'issue_time', 'pm25']].copy()
    target.issue_time -= pd.Timedelta(hours=24)
    df = df.merge(target.rename(columns={'pm25': 'target_pm25'}), on=['location_id', 'issue_time'], how='left', validate='one_to_one')
    df['target_time'] = df.issue_time + pd.Timedelta(hours=24)
    # Weekly seasonal forecast y(t+24-168) = y(t-144).
    weekly = hours[['location_id', 'issue_time', 'pm25']].copy()
    weekly.issue_time += pd.Timedelta(hours=144)
    df = df.merge(weekly.rename(columns={'pm25': 'weekly_persistence'}), on=['location_id', 'issue_time'], how='left', validate='one_to_one')
    local = df.target_time.dt.tz_convert('Asia/Kolkata')
    df['target_hour'] = local.dt.hour
    df['target_weekday'] = local.dt.dayofweek
    df['target_month'] = local.dt.month
    return df.dropna(subset=['pm25', 'target_pm25']).sort_values(['issue_time', 'location_id']).reset_index(drop=True)


def chronological_split(df):
    times = df.issue_time.drop_duplicates().sort_values().tolist()
    if len(times) < 20:
        raise ValueError('Need at least 20 distinct issue times')
    calibration_start, test_start = times[int(len(times)*.70)], times[int(len(times)*.85)]
    # Purge labels crossing either boundary; random row splits are forbidden.
    train = df[df.target_time < calibration_start]
    calibration = df[(df.issue_time >= calibration_start) & (df.target_time < test_start)]
    test = df[df.issue_time >= test_start]
    if any(part.empty for part in (train, calibration, test)):
        raise ValueError('Insufficient temporal coverage after 24h boundary purge')
    return train, calibration, test


def available_forecasts(frame):
    """Validate a normalized weather/CAMS archive before joining to features."""
    out = frame.copy()
    for column in ['issue_time', 'available_at', 'valid_time']:
        out[column] = pd.to_datetime(out[column], utc=True, errors='raise')
    if out[['issue_time', 'available_at', 'valid_time']].isna().any().any():
        raise ValueError('Forecast timestamps must be present')
    if (out.available_at > out.issue_time).any():
        raise ValueError('Forecast not available at prediction issue time')
    if (out.valid_time != out.issue_time + pd.Timedelta(hours=24)).any():
        raise ValueError('Forecast must verify exactly 24 hours later')
    return out


def rolling_splits(df, calibration_days=28, test_days=28, minimum_train_days=90):
    """Expanding-window evaluation; disjoint test issue windows, purged labels.

    Earlier test labels may enter later training only once observed. These are
    development folds, not a fresh holdout for selecting hyperparameters.
    """
    if min(calibration_days, test_days, minimum_train_days) < 1:
        raise ValueError('Window lengths must be positive')
    day = pd.Timedelta(days=1)
    start = df.issue_time.min().ceil('D') + (minimum_train_days+calibration_days)*day
    while start + test_days*day <= df.issue_time.max():
        cal_start = start-calibration_days*day
        train = df[df.target_time < cal_start]
        cal = df[(df.issue_time >= cal_start) & (df.target_time < start)]
        test = df[(df.issue_time >= start) & (df.issue_time < start+test_days*day)]
        if not any(part.empty for part in (train,cal,test)):
            yield train, cal, test
        start += test_days*day
