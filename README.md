# AeroShield IQ

**Transforming raw air quality data into proactive, actionable city intelligence.**

> **v3 research upgrade:** AeroShield now uses chronological evaluation,
> conformal uncertainty bands, 600-run uncertainty-aware plume screening,
> live forecast meteorology, conservative intervention triage,
> and a human-approval gate. See
> [`docs/RESEARCH_UPGRADE.md`](docs/RESEARCH_UPGRADE.md) for the research basis,
> architecture, benchmark protocol, and honest limitations.

Urban air pollution is a severe public health crisis, yet city authorities often rely on
reactive advisories due to a lack of actionable intelligence. AeroShield IQ fuses open
ground-sensor data, meteorological inputs, and ML to estimate hyperlocal pollution,
rank inventoried source hypotheses via reverse-plume dispersion modelling, and draft
field-verification briefs for human review.

This README covers: what's implemented, how the pieces fit together, how to run it end
to end, every API endpoint, and known limitations/what's next.

---

## 1. What's implemented

| Layer | What it does |
|---|---|
| **Data pipeline** (`pipeline/step1-3`) | Downloads a year of Delhi CPCB/DPCC sensor readings from OpenAQ, builds spatial layers (OSM road density, industrial/construction emission sources), joins Open-Meteo weather, engineers features, and trains a LightGBM surrogate model. |
| **Conditional estimation** (`app/ml/inference.py`) | Serves PM2.5 scenario estimates across a 15×15 Delhi grid or at a clicked point. This is not yet a validated multi-horizon forecast. |
| **Source screening** (`app/ml/plume_math.py`) | Reverse Gaussian-plume ensemble ranks relative contribution hypotheses among the incomplete demonstration inventory. Shares are not calibrated source probabilities. |
| **Uncertainty calibration** (`app/ml/uncertainty.py`) | Quantile estimates plus split-conformal calibration produce auditable nominal 90% PM2.5 bands. |
| **Forecast weather** (`app/ml/weather.py`) | Loads and caches Open-Meteo wind and boundary-layer forecasts, with a visible fallback state. |
| **Live operating context** (`app/live_context.py`) | Loads current PM2.5 plus a coarse 24-hour Open-Meteo/CAMS baseline, with memory and disk-cache states shown explicitly. |
| **Satellite evidence** (`app/satellite.py`) | Loads NASA FIRMS VIIRS thermal anomalies and ranks them by distance and wind compatibility; requires `NASA_FIRMS_MAP_KEY` for live data. |
| **Health advisories** (`app/health_advisory.py`) | Generates deterministic, reviewable English/Hindi messages for the public, sensitive groups, schools, and outdoor workers from the conservative interval bound. |
| **Intervention optimizer** (`app/ml/intervention.py`) | Ranks field actions by a conservative, non-causal benefit proxy, cost, and response time; efficacy constants are scenario assumptions. |
| **Verification agent** (`app/agents/orchestrator.py`) | A 2-node LangGraph pipeline (Planner → Drafter) backed by Groq/Llama-3 that drafts a field-verification brief. It falls back to conservative deterministic wording without `GROQ_API_KEY`. |
| **Case persistence** (`app/db.py`) *(new)* | SQLite log of every dispatched enforcement case — survives page refresh/restart. |
| **Dashboard** (`frontend/src/App.jsx`) | React + Leaflet map: live PM2.5 heatmap, wind vector, sensor/source markers, date+hour forecast controls, and the enforcement sidebar. |

---

## 2. Architecture

```
                 ┌─────────────────────┐
 OpenAQ API ───▶ │ step1_download       │
                 │ _openaq.py           │
                 └──────────┬───────────┘
                            ▼
                 ┌─────────────────────┐
 OSM / OSMnx ──▶ │ step2_spatial        │
                 │ _layers.py           │──▶ delhi_emission_sources.json
                 └──────────┬───────────┘    delhi_grid_road_density.csv
                            ▼
                 ┌─────────────────────┐
 Open-Meteo ───▶ │ step3_etl_and       │
                 │ _train.py            │──▶ surrogate_model.joblib
                 └──────────┬───────────┘    model_meta.json (features + metrics)
                            ▼
        ┌───────────────────────────────────────┐
        │              FastAPI (main.py)          │
        │  /api/city-grid    /api/analyze-hotspot  │
        │  /api/model-info   /api/dispatch         │
        │  /api/cases                              │
        │        │                │                │
        │   inference.py    plume_math.py           │
        │        │                │                │
        │        │          orchestrator.py         │
        │        │           (LangGraph)             │
        │        │                │                │
        │        ▼                ▼                │
        │            aeroshield.db (SQLite)         │
        └───────────────────┬───────────────────────┘
                             ▼
                  React + Leaflet dashboard
```

---

## 3. Project structure

```text
aeroshield-iq/
├── backend/
│   ├── .env                        # API keys (OpenAQ, Groq) — you create this
│   ├── main.py                     # FastAPI server & all endpoints
│   ├── app/
│   │   ├── db.py                   # (new) SQLite persistence for dispatched cases
│   │   ├── agents/
│   │   │   └── orchestrator.py     # LangGraph 2-agent enforcement pipeline
│   │   └── ml/
│   │       ├── inference.py        # Grid + single-point prediction
│   │       ├── plume_math.py       # Reverse Gaussian-plume source attribution
│   │       └── train.py            # synthetic Bangalore demo trainer — NOT the real pipeline
│   └── pipeline/
│       ├── step1_download_openaq.py
│       ├── step2_spatial_layers.py
│       └── step3_etl_and_train.py
├── frontend/
│   ├── package.json
│   └── src/
│       └── App.jsx                 # Dashboard: map, forecast controls, enforcement sidebar
└── data/                            # Auto-generated — models, grids, JSONs, aeroshield.db
```

---

## 4. Setup

### Prerequisites
- Node.js v18+
- Python 3.10+
- Free API keys: [OpenAQ](https://explore.openaq.org) and [Groq](https://console.groq.com) (Groq is optional — the enforcement agent has a rule-based fallback)

### 4.1 Environment variables

```bash
cd aeroshield-iq/backend
cat > .env <<'EOF'
OPENAQ_API_KEY="your_openaq_key"
GROQ_API_KEY="your_groq_key"
EOF
```

### 4.2 Install backend dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn lightgbm requests python-dotenv \
            langchain-groq langgraph osmnx shapely tqdm joblib
```

### 4.3 Install frontend dependencies

```bash
cd ../frontend
npm install
npm install leaflet react-leaflet axios
```

---

## 5. Running the data pipeline (required before first launch)

Run these **in order** from `backend/`. This is what actually produces
`surrogate_model.joblib`, `delhi_emission_sources.json`,
`delhi_grid_road_density.csv`, and `model_meta.json` in `/data`.

```bash
cd backend
python pipeline/step1_download_openaq.py     # ~10-15 min — pulls a year of OpenAQ readings
python pipeline/step2_spatial_layers.py      # builds the road-density grid + emission source list
python pipeline/step3_etl_and_train.py       # ETL + trains the LightGBM model, logs RMSE/MAE
```

`step3` now prints and saves validation metrics, e.g.:
```
[step3] Validation — RMSE: 11.84 µg/m³  MAE: 8.02 µg/m³  Within ±15 µg/m³: 78.3%
```
These are what power the "Model: RMSE ... MAE ..." badge in the dashboard footer.

> **Important:** `app/ml/train.py` is a separate, synthetic demo-data trainer built
> around Bangalore coordinates — it is *not* wired into the app automatically anymore.
> If `surrogate_model.joblib` is missing, the backend will now fail loudly with a clear
> error instead of silently falling back to it. Always run the three pipeline steps
> above for real Delhi predictions.

---

## 6. Running the app

**Backend** (from `backend/`):
```bash
python main.py
```
This also creates `data/aeroshield.db` (SQLite) on first startup via `db.init_db()`.
Server runs at `http://127.0.0.1:8000`.

**Frontend** (from `frontend/`, in a new terminal):
```bash
npm run dev
```
Open `http://localhost:5173`.

---

## 7. Using the dashboard

1. Pick a **date and hour** at the top — the model refits its seasonal (`month`) and
   weekday (`day_of_week`) features accordingly, so a winter weekday and a monsoon
   Sunday will show different pollution patterns even at the same hour.
2. Click **⚡ Forecast** (or just release the slider/change the date) to refresh the grid.
3. Click any **orange/red hotspot cell** to run 600-simulation reverse-plume attribution — the
   sidebar shows the PM2.5 uncertainty band, source rank stability, and robust intervention options.
4. Click **✓ Approve Field Verification** — this POSTs to `/api/dispatch` and
   saves the case to SQLite. Refreshing the page won't lose it; query it back via
   `GET /api/cases`.

The generated brief is decision support for a human officer. It is not a legal
finding and must not be used as the sole basis for enforcement.

---

## 8. What's new in this version

### Feature 1 — Persisted enforcement cases (SQLite)
- New `app/db.py` module, table `enforcement_cases` in `data/aeroshield.db`.
- New endpoints: `POST /api/dispatch` (saves a case, returns `case_id`) and
  `GET /api/cases` (lists recent cases, most recent first).
- The frontend's dispatch button now does a real network round-trip instead of a fake
  1.8s `setTimeout`, and shows the real `case_id` returned by the backend.

### Feature 2 — Model accuracy badge
- `step3_etl_and_train.py` now computes RMSE, MAE, and "% of predictions within
  ±15 µg/m³ of actual" on the held-out validation split, and saves them into
  `model_meta.json`.
- New endpoint `GET /api/model-info` surfaces this.
- The dashboard footer now shows e.g. `Model: RMSE 11.8 · MAE 8.0 · ±15µg/m³ 78.3%`.

### Feature 3 — Date + hour conditional scenario
- The header now has a date picker next to the hour slider.
- `day_of_week` and `month` are derived from the picked date and sent to
  `GET /api/city-grid`, so seasonal effects (e.g. winter stubble-burning peaks vs.
  monsoon washout) and weekday/weekend traffic patterns shift the estimate.
- Open-Meteo supplies forecast wind and boundary-layer height when available; the API
  exposes a fallback note otherwise. Because fixed/default lag values are used and no
  direct future-horizon targets were validated, this remains a conditional scenario,
  not a validated multi-day PM2.5 forecast.

---

## 9. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/city-grid?hour=&day_of_week=&month=&wind_speed=&wind_direction=` | 15×15 grid of PM2.5 predictions + AQI category per cell |
| `POST` | `/api/analyze-hotspot` | Runs plume attribution + real point prediction + enforcement agent for one lat/lon |
| `GET` | `/api/model-info` | Returns `model_meta.json` (features, grid config, validation metrics) |
| `GET` | `/api/live-context` | Current/coarse external PM2.5 context plus a 24-hour outlook and cache status |
| `GET` | `/api/satellite-fires` | NASA FIRMS fire evidence or an explicit not-configured/cache state |
| `POST` | `/api/health-advisory` | Deterministic English/Hindi group-specific health-risk messages |
| `POST` | `/api/dispatch` | Persists a dispatched case to SQLite, returns `case_id` |
| `GET` | `/api/cases?limit=50` | Lists recently dispatched cases |

`POST /api/analyze-hotspot` body:
```json
{
  "cell_id": 112,
  "lat": 28.630, "lon": 77.210,
  "wind_speed": 4.0, "wind_direction": 315.0,
  "hour": 21, "day_of_week": 3
}
```

`POST /api/dispatch` body: same shape as the `/api/analyze-hotspot` response, flattened
(see `App.jsx::handleDispatch` for the exact fields sent).

---

## 10. Known limitations / honest next steps

- Open-Meteo weather is live when available, but the PM2.5 estimator has not been
  trained for explicit 1–72 hour targets. Direct horizon labels and horizon-specific
  evaluation are required before claiming multi-hour forecasting.
- The new 24-hour Open-Meteo/CAMS outlook is displayed only as a coarse external
  operational baseline. It is not relabelled as AeroShield's hyperlocal forecast.
- The source inventory is demonstration data and may omit real emitters; attribution
  outputs are conditional relative shares, not calibrated probabilities.
- Intervention efficacy values are unvalidated scenario assumptions; displayed
  benefit values are prioritization proxies, not causal reductions.
- **`train.py`'s Bangalore synthetic model** is dead weight kept only as a quick
  local-dev sanity trainer — consider deleting it or rewriting it to match the real
  Delhi feature schema if you still want a synthetic fallback.
- **No authentication** is implemented. CORS defaults to the two local Vite
  origins and can be configured with `AEROSHIELD_CORS_ORIGINS`; authentication
  and authorization are still required before any real deployment.
- `app/ml/train.py` still contains legacy synthetic feature logic and must not be used
  to produce release artifacts.

Before merging, complete [`docs/VALIDATION_CHECKLIST.md`](docs/VALIDATION_CHECKLIST.md).

### Optional live satellite setup

Register for a NASA FIRMS MAP_KEY, then set it before starting the backend:

```bash
export NASA_FIRMS_MAP_KEY="your-key"
```

Without a key, `/api/satellite-fires` returns `mode=not_configured` (or the last
valid disk cache). AeroShield never fabricates satellite detections for a demo.

---

## Hosted demo: Vercel frontend + Render API

Keep the browser and secret-bearing integrations separate:

1. Create the Render service from `render.yaml`. In Render, set
   `NASA_FIRMS_MAP_KEY` as a secret and set `AEROSHIELD_CORS_ORIGINS` to the exact
   production Vercel origin (for example, `https://aeroshield.vercel.app`). Do not
   expose the FIRMS key as a `VITE_*` variable.
2. In Vercel, import this repository with **Root Directory** set to `frontend`.
   Set `VITE_API_BASE_URL` to the Render service origin, without a trailing slash
   (for example, `https://aeroshield-api.onrender.com`).
3. Deploy Render first, confirm `GET /api/health` reports
   `nasa_firms_configured: true`, and then deploy Vercel.
4. Confirm `/api/satellite-fires` reports `mode: live` or an explicitly labelled
   cached/unavailable state. The health endpoint exposes configuration booleans
   only; it never returns secret values.

Render's default filesystem is ephemeral, so the SQLite case log can reset after
a redeploy or service restart. This is acceptable for the one-week demo but must
be replaced with a managed database before claiming durable operational history.

## License

MIT License.
