"""Immutable prospective input capture; run from backend with --help.

Collection is separate from serving. Failure must never create a scenario forecast
in this archive. Store --output on durable storage in hosted deployments.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from uuid import uuid4


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone is required')
    return result.astimezone(timezone.utc)


def capture(provider, url, output, headers=None):
    started = datetime.now(timezone.utc).isoformat()
    with urlopen(Request(url, headers=headers or {}), timeout=30) as response:
        raw = response.read()
    received = datetime.now(timezone.utc).isoformat()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get('error'):
        raise ValueError('Provider returned an invalid/error payload')
    envelope = {'schema_version': 1, 'provider': provider, 'source_url': url,
                'requested_at': started, 'available_at': received,
                'model_run_at': None, 'availability_basis': 'response_received',
                'raw_sha256': hashlib.sha256(raw).hexdigest(), 'payload': payload}
    directory = Path(output) / provider / received[:10]
    directory.mkdir(parents=True, exist_ok=True)
    stem = received.replace(':', '-') + '-' + uuid4().hex
    # Raw bytes preserve the exact response for hash verification. Never overwrite.
    (directory / (stem + '.raw')).write_bytes(raw)
    path = directory / (stem + '.json')
    with path.open('x') as handle:
        json.dump(envelope, handle, allow_nan=False)
    return path


def forecast_at(snapshot, issue_time, variable):
    """Select exact t+24 from a snapshot actually received by issue time.

    Open-Meteo requests below explicitly use GMT. Returned timestamps denote
    valid time, never model initialization or historical availability.
    """
    issue = utc(issue_time)
    if utc(snapshot['available_at']) > issue:
        raise ValueError('Forecast was unavailable at prediction time')
    if snapshot['provider'] not in ('weather', 'cams'):
        raise ValueError('Not a forecast snapshot')
    payload = snapshot['payload']
    if payload.get('utc_offset_seconds') != 0:
        raise ValueError('Expected explicit UTC forecast')
    hourly = payload['hourly']
    target = issue + timedelta(hours=24)
    for index, time in enumerate(hourly['time']):
        valid = utc(time + '+00:00') if len(time) == 16 else utc(time)
        if valid == target:
            value = hourly[variable][index]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Missing/nonfinite forecast')
            return value
    raise ValueError('Exact target hour absent; no nearest-hour substitution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('provider', choices=['weather', 'weather_history', 'cams', 'openaq'])
    parser.add_argument('--latitude', type=float, default=28.6139)
    parser.add_argument('--longitude', type=float, default=77.2090)
    parser.add_argument('--sensor-id', type=int)
    parser.add_argument('--start-date', help='Historical weather start, YYYY-MM-DD')
    parser.add_argument('--end-date', help='Historical weather end, YYYY-MM-DD')
    parser.add_argument('--output', default='../data/prospective')
    args = parser.parse_args()
    headers = {}
    params = {'latitude': args.latitude, 'longitude': args.longitude, 'timezone': 'GMT', 'forecast_days': 5}
    if args.provider == 'weather':
        endpoint = 'https://api.open-meteo.com/v1/forecast'
        params.update(models='gfs_global', wind_speed_unit='ms', hourly='temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation')
    elif args.provider == 'weather_history':
        if not args.start_date or not args.end_date:
            parser.error('Historical weather requires --start-date and --end-date')
        endpoint = 'https://previous-runs-api.open-meteo.com/v1/forecast'
        params.pop('forecast_days')
        params.update(start_date=args.start_date, end_date=args.end_date, models='gfs_global', wind_speed_unit='ms',
            hourly=','.join(name + '_previous_day1' for name in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'precipitation']))
    elif args.provider == 'cams':
        endpoint = 'https://air-quality-api.open-meteo.com/v1/air-quality'
        params.update(domains='cams_global', hourly='pm2_5')
    else:
        if not args.sensor_id or not os.environ.get('OPENAQ_API_KEY'):
            parser.error('OpenAQ needs --sensor-id and OPENAQ_API_KEY in the environment')
        endpoint = f'https://api.openaq.org/v3/sensors/{args.sensor_id}/measurements'
        now = datetime.now(timezone.utc)
        params = {'limit': 1000, 'page': 1, 'date_from': (now-timedelta(days=1)).isoformat(), 'date_to': now.isoformat()}
        headers = {'X-API-Key': os.environ['OPENAQ_API_KEY']}
        # A bounded raw capture, not a complete historical backfill. Keep meta
        # pagination and source quality fields; downstream must check completeness.
    print(capture(args.provider, endpoint + '?' + urlencode(params), args.output, headers))


if __name__ == '__main__':
    main()
