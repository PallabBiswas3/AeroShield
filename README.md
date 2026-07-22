# AeroShield IQ

**Transforming raw air quality data into proactive, actionable city intelligence.**

Urban air pollution is a severe public health crisis, yet city authorities often rely on
reactive advisories due to a lack of actionable intelligence. AeroShield IQ fuses open
ground-sensor data, meteorological baselines, and AI to forecast hyperlocal pollution,
pinpoint industrial culprits via reverse-plume dispersion modelling, and auto-draft
enforcement mandates for review and dispatch.

This README covers: what's implemented, how the pieces fit together, how to run it end
to end, every API endpoint, and known limitations/what's next.

---

## 1. What's implemented

| Layer | What it does |
|---|---|
| **Data pipeline** (`pipeline/step1-3`) | Downloads a year of Delhi CPCB/DPCC sensor readings from OpenAQ, builds spatial layers (OSM road density, industrial/construction emission sources), joins Open-Meteo weather, engineers features, and trains a LightGBM surrogate model. |
| **Forecasting** (`app/ml/inference.py`) | Serves PM2.5 predictions across a 15×15 grid over Delhi NCT, or for a single exact lat/lon (used when you click a hotspot). |
| **Source attribution** (`app/ml/plume_math.py`) | Reverse Gaussian-plume dispersion model — given a hotspot and wind vector, ranks nearby industrial/construction sources by how likely each is to be the cause. |
| **Enforcement agent** (`app/agents/orchestrator.py`) | A 2-node LangGraph pipeline (Planner → Legal Drafter) backed by Groq/Llama-3 that triages severity against Indian environmental statutes and drafts a legal notice. Falls back to a deterministic rule-based version if no `GROQ_API_KEY` is set. |
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
│   │       └── train.py            # ⚠️ synthetic Bangalore demo trainer — NOT the real pipeline
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
3. Click any **orange/red hotspot cell** to run reverse-plume attribution — the
   sidebar shows the model's real predicted PM2.5 for that exact point, the ranked list
   of probable industrial sources, and an AI-drafted enforcement mandate.
4. Click **✓ Approve & Dispatch Field Squad** — this now POSTs to `/api/dispatch` and
   saves the case to SQLite. Refreshing the page won't lose it; query it back via
   `GET /api/cases`.

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

### Feature 3 — Date + hour forecast
- The header now has a date picker next to the hour slider.
- `day_of_week` and `month` are derived from the picked date and sent to
  `GET /api/city-grid`, so seasonal effects (e.g. winter stubble-burning peaks vs.
  monsoon washout) and weekday/weekend traffic patterns actually shift the forecast.
- **Honest caveat:** wind speed/direction are still a manually-set baseline, not a real
  multi-day weather forecast — picking a date 3 days out changes the model's seasonal
  and weekday features correctly, but doesn't fetch a genuine future wind forecast for
  that date. See §10 for what a full fix would need.

---

## 9. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/city-grid?hour=&day_of_week=&month=&wind_speed=&wind_direction=` | 15×15 grid of PM2.5 predictions + AQI category per cell |
| `POST` | `/api/analyze-hotspot` | Runs plume attribution + real point prediction + enforcement agent for one lat/lon |
| `GET` | `/api/model-info` | Returns `model_meta.json` (features, grid config, validation metrics) |
| `POST` | `/api/dispatch` | Persists a dispatched case to SQLite, returns `case_id` |
| `GET` | `/api/cases?limit=50` | Lists recently dispatched cases |

`POST /api/analyze-hotspot` body:
```json
{
  "cell_id": 112,
  "lat": 28.735, "lon": 77.15,
  "wind_speed": 4.0, "wind_direction": 315.0,
  "hour": 21, "day_of_week": 3
}
```

`POST /api/dispatch` body: same shape as the `/api/analyze-hotspot` response, flattened
(see `App.jsx::handleDispatch` for the exact fields sent).

---

## 10. Known limitations / honest next steps

- **Wind is still a manual baseline**, not pulled from a real forecast for the selected
  future date. A proper fix: call Open-Meteo's *forecast* (not archive) endpoint for the
  picked date and feed real predicted wind into `/api/city-grid`.
- **`train.py`'s Bangalore synthetic model** is dead weight kept only as a quick
  local-dev sanity trainer — consider deleting it or rewriting it to match the real
  Delhi feature schema if you still want a synthetic fallback.
- **No auth** on any endpoint and `CORS` is wide open (`allow_origins=["*"]`) — fine
  for local dev, must be locked down before any real deployment.
- **`/api/cases` has no UI yet** — the data is there (and query-able), but there's no
  "case log" panel in the dashboard to browse past dispatches.
- The three duplicated implementations of `traffic_density`/diurnal-factor logic
  (in `train.py`, `step3_etl_and_train.py`, and `inference.py`) still aren't
  consolidated into one shared module — flagged previously, not yet addressed.

---

## License
MIT License.