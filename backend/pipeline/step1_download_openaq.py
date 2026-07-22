# backend/pipeline/step1_download_openaq.py
import os, time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "YOUR_KEY_HERE")
DATE_FROM = "2025-01-01"
DATE_TO   = "2025-12-31"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_openaq"))

DELHI_STATIONS = [
    {"openaq_id": 17, "name": "R K Puram, Delhi - DPCC", "agency": "DPCC", "lat": 28.5630, "lon": 77.1870, "zone": "South", "note": "Has embedded wind sensor — best station"},
    {"openaq_id": 5404, "name": "Pusa, Delhi - IMD", "agency": "IMD", "lat": 28.6396, "lon": 77.1463, "zone": "Central-West", "note": "No wind — will use Open-Meteo"},
    {"openaq_id": 6358, "name": "Mandir Marg, New Delhi - DPCC", "agency": "DPCC", "lat": 28.6364, "lon": 77.2011, "zone": "Central", "note": "Has embedded wind sensor"},
    {"openaq_id": 8118, "name": "New Delhi - DPCC", "agency": "DPCC", "lat": 28.6139, "lon": 77.2090, "zone": "Central", "note": "OpenAQ quickstart example station"},
    {"openaq_id": 8119, "name": "Anand Vihar, Delhi - DPCC", "agency": "DPCC", "lat": 28.6508, "lon": 77.3152, "zone": "East", "note": "Critical — near Anand Vihar bus terminal, very high PM2.5"},
    {"openaq_id": 8120, "name": "Punjabi Bagh, Delhi - DPCC", "agency": "DPCC", "lat": 28.6726, "lon": 77.1321, "zone": "West", "note": "High traffic area"},
    {"openaq_id": 145, "name": "Jahangirpuri, Delhi - DPCC", "agency": "DPCC", "lat": 28.7290, "lon": 77.1676, "zone": "North", "note": "Near Bawana industrial cluster"},
    {"openaq_id": 1437, "name": "Rohini, Delhi - DPCC", "agency": "DPCC", "lat": 28.7041, "lon": 77.1164, "zone": "North-West", "note": "Covers Narela/Bawana downwind"},
    {"openaq_id": 1439, "name": "Narela, Delhi - DPCC", "agency": "DPCC", "lat": 28.8550, "lon": 77.0935, "zone": "North", "note": "Closest station to Narela industrial estate"},
    {"openaq_id": 1440, "name": "Bawana, Delhi - DPCC", "agency": "DPCC", "lat": 28.7705, "lon": 77.0398, "zone": "North-West", "note": "Inside Bawana industrial zone — key culprit station"},
    {"openaq_id": 1443, "name": "Okhla Phase-2, Delhi - DPCC", "agency": "DPCC", "lat": 28.5355, "lon": 77.2781, "zone": "South-East", "note": "Inside Okhla industrial area"},
    {"openaq_id": 1444, "name": "Wazirpur, Delhi - DPCC", "agency": "DPCC", "lat": 28.7041, "lon": 77.1726, "zone": "North-West", "note": "Inside Wazirpur industrial cluster"},
]

def download_station(station: dict, date_from: str, date_to: str) -> pd.DataFrame:
    import requests
    loc_id = station["openaq_id"]
    headers = {"X-API-Key": OPENAQ_API_KEY}
    
    loc_url = f"https://api.openaq.org/v3/locations/{loc_id}"
    try:
        resp = requests.get(loc_url, headers=headers, timeout=10)
        if resp.status_code != 200: return pd.DataFrame()
        sensors = resp.json().get("results", [])[0].get("sensors", [])
    except Exception: return pd.DataFrame()
        
    if not sensors: return pd.DataFrame()
    all_rows = []
    
    for sensor in sensors:
        sensor_id = sensor["id"]
        param_name = sensor.get("parameter", {}).get("name", "")
        param_unit = sensor.get("parameter", {}).get("units", "")
        meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
        params = {"datetime_from": f"{date_from}T00:00:00Z", "datetime_to": f"{date_to}T23:59:59Z", "limit": 1000, "page": 1}
        
        while True:
            try:
                m_resp = requests.get(meas_url, headers=headers, params=params, timeout=30)
                if m_resp.status_code != 200: break
                results = m_resp.json().get("results", [])
                if not results: break
                    
                for r in results:
                    period_data = r.get("period", {})
                    all_rows.append({"location_id": loc_id, "location_name": station["name"], "parameter": param_name, "value": r.get("value", None), "unit": param_unit, "datetimeUtc": period_data.get("datetimeFrom", {}).get("utc", ""), "datetimeLocal": period_data.get("datetimeFrom", {}).get("local", ""), "timezone": "Asia/Kolkata", "latitude": station["lat"], "longitude": station["lon"]})
                    
                if len(results) < params["limit"]: break
                params["page"] += 1
                time.sleep(0.3)
            except Exception: break
        time.sleep(0.3)
    return pd.DataFrame(all_rows)

def main():
    if OPENAQ_API_KEY == "YOUR_KEY_HERE": return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []
    for station in tqdm(DELHI_STATIONS, desc="Downloading stations"):
        try:
            df = download_station(station, DATE_FROM, DATE_TO)
            if len(df) == 0: continue
            out_path = os.path.join(OUTPUT_DIR, f"openaq_location_{station['openaq_id']}_2025.csv")
            df.to_csv(out_path, index=False)
            all_frames.append(df)
        except Exception: continue
        time.sleep(1.0)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "delhi_all_stations_2025.csv"), index=False)

if __name__ == "__main__": main()