# Data and validation implementation register

Research/access checks: 2026-09-13. This is an implementation plan with explicit gates,
not evidence that the proposed integrations or scientific validation are complete.

## Forecast inputs

| Layer | Official source | Implementation and gate |
|---|---|---|
| Monitoring | [OpenAQ v3 measurements](https://docs.openaq.org/resources/measurements), [API key](https://docs.openaq.org/using-the-api/api-key) | Raw capture CLI implemented. Requires OPENAQ_API_KEY and verified Delhi PM2.5 sensor IDs. Capture sensor/location/provider/license metadata too. Preserve period start/end, retrieval time and upstream flags; normalize micrograms/m³; quarantine invalid/negative/duplicate records. Hourly completeness and pagination must be checked before ingestion into training. Never impute target values. |
| Issued historical weather | [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api), [Single Runs](https://open-meteo.com/en/docs/single-runs-api) | Fixed-lead GFS download implemented and 48 hours successfully captured for Sep 1–2, 2025. Previous-day1 means fixed 24-hour lead relative to valid time; it is not proof of exact dissemination time. For strict backtests retain run initialization plus conservative publication delay, or prove availability from prospective snapshots. Do not use reanalysis as issued forecasts. Full backfill and feature joins remain pending. |
| Prospective meteorology | [Open-Meteo forecast](https://open-meteo.com/en/docs) | GFS global capture implemented: temperature, RH, precipitation, wind speed m/s and meteorological direction. Raw bytes and receipt timestamp retained. Successful live capture in this workspace; this does not verify Render's network access. |
| CAMS baseline | [Open-Meteo air quality](https://open-meteo.com/en/docs/air-quality-api) | Explicit cams_global PM2.5 capture implemented and live request succeeded. Roughly 45 km global model output, not a monitoring observation or validated 1 km truth. Record receipt time separately from unknown initialization. Compare against the same station targets, hours, units and missingness mask. Prospective evaluation needs targets collected after forecasts mature. |

Run from backend:

```bash
python -m pipeline.archive_inputs weather --output /durable/path/aeroshield
python -m pipeline.archive_inputs cams --output /durable/path/aeroshield
python -m pipeline.archive_inputs weather_history --start-date 2025-09-01 --end-date 2025-09-02 --output /durable/path/aeroshield
python -m pipeline.archive_inputs openaq --sensor-id VERIFIED_SENSOR_ID --output /durable/path/aeroshield
```

Set OPENAQ_API_KEY through the deployment secret manager, never in source control.
Current OpenAQ command is a bounded first-page raw capture, not complete ingestion.
Schedule captures per station on durable storage outside request handling; collector
is not yet scheduled. Local data/prospective is ignored by Git and is not durable.
Copy envelopes AND raw files to object storage before relying on the archive.
Three small initial snapshots are preserved in data/evidence_snapshots in Git for
reproducible access checks: CAMS 120 hours, GFS 120 hours, archived GFS 48 hours.
They are frozen evidence fixtures, not a continuously updated feed.
Use forecast_at only with a snapshot received before the issue time and an exact
t+24 valid hour. Never backdate receipt time or fill failed downloads with scenarios.

Production weather already has a cooldown and explicit scenario fallback. This
change also fixes cached responses falsely receiving a fresh fetched_at timestamp.
Provider failures remain possible; monitor freshness and error rate from Render.

## Intervention evidence

| Layer | Source | Concrete next implementation |
|---|---|---|
| Inventory | [CPCB-hosted ARAI/TERI August 2018 report](https://prana.cpcb.gov.in/ncapDashboard/downloadEmissionFiles/Delhi.pdf), [TERI summary](https://www.teriin.org/sites/default/files/2018-08/Exec-summary.pdf) | Extract cited tables with page/table IDs, geography, pollutant, season, year and method. Map shared categories explicitly, preserving secondary aerosol/regional background. Compare aggregate category ranks within matched scope. Do not label inventory agreement as per-cell top-1 attribution accuracy. |
| Schools and hospitals | [OSM Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API) | Snapshot nodes/ways/relations with amenity=school/hospital inside the demo bounds; retain OSM IDs, tags, timestamp and attribution. Deduplicate footprints and entrances, manually check sample locations. Coverage is incomplete; facility counts are not patient/pupil counts. |
| Wards | [Delhi SEC maps](https://sec.delhi.gov.in/maps) | Obtain licensed machine-readable polygons matched to the selected delimitation year; verify against official maps. Search results contain older ward schemes, so do not silently mix years. Validate CRS, geometry, ward IDs and overlaps. Verified current vector dataset remains unresolved. |
| Population | [WorldPop data hub](https://hub.worldpop.org/) | Pin an India population-count raster record, year, resolution and license. Sum fractional raster overlap by ward, preserving NoData. Do not sum density as counts or duplicate residents across overlapping wards. Dataset selection/download and exposure computation remain pending. |

Unresolved-source behavior: when evidence is missing, mismatched in time, diffuse,
or rankings are unstable to plausible wind changes, return unresolved/diffuse with
the reason. Do not invent confidence percentages or use a probability threshold
until its calibration has been measured. Rank recommendations as investigation
priorities, not guilt, causal source proof or guaranteed pollution reduction.

Exposure: report residents within forecast exceedance areas with separate lower,
central and upper concentration scenarios. These are exposure scenarios, not a
population confidence interval. Keep school/hospital counts separate from residents.

## Real operator and expert study (not conducted)

Recruit one environmental scientist or air-quality practitioner and at least one
intended operator. Prepare 12 frozen cases spanning high/low pollution, missing
weather and unresolved attribution. Randomize/counterbalance baseline dashboard
versus AeroShield cases to reduce learning effects. Record signal_available_at,
task_start, recommendation_ready and operator_decision separately. Measure end-to-end
signal-to-recommendation and active operator time; report median, tail, failures,
sample size and protocol, not an API latency proxy for an actual operator study.

Expert independently scores evidence sufficiency, feasible action, priority order,
uncertainty communication and harmful overreach (1–5 with written rationale).
Preserve disagreements and conflicts of interest. No expert has been contacted and
no review score or time saving may be claimed yet. Outreach needs a named recipient
and authorization before sending messages.

## Model strengthening and promotion checklist

- Freeze a fresh evaluation period before further model selection. Existing eight
  windows are development data now; do not call them untouched validation.
- Compare residual LightGBM, Ridge, persistence and weekly persistence on identical
  eligible targets. Add CAMS only after issue-time availability is established.
- Add weather features as an ablation; publish station/window RMSE, MAE, skill,
  sample counts, coverage and interval width, including negative-skill windows.
- Evaluate missing-input periods, pollution episodes and held-out stations. Station
  forecasting performance cannot establish 1 km spatial forecast accuracy.
- Choose fallback/blending using only earlier selection data; keep calibration
  separate. Test a simple regularized blend before any complex architecture.
- Evaluate delayed-feedback intervals with realistic reporting latency, not just
  target completion time. Report coverage by station/window and concentration band.
- Add moving-block uncertainty estimates for paired forecast errors; hourly samples
  are dependent. Poor subgroup coverage must remain visible despite pooled success.
- Publish data hashes, model parameters, dependency versions, split boundaries,
  source manifests and full machine-readable results; retain raw snapshots durably.
- Promote only after fresh evidence supports improvement, acceptable interval width
  and reliability. Keep the live estimator's conditional label until then.

## Additional model experiment

Ridge(alpha=10), training-only imputation/scaling and categorical station/calendar
encoding, residual targets and delayed station calibration were evaluated using:

```bash
python -m pipeline.benchmark_forecast24 --rolling --model ridge --residual --adaptive
```

| Window start (2025) | RMSE skill vs persistence | 90% interval coverage |
|---|---:|---:|
| Apr 29 | 15.0% | 91.7% |
| May 27 | 0.1% | 91.4% |
| Jun 24 | 2.7% | 89.3% |
| Jul 22 | -43.4% | 90.4% |
| Aug 19 | -28.2% | 89.9% |
| Sep 16 | -29.3% | 89.3% |
| Oct 14 | -56.1% | 90.5% |
| Nov 11 | -5.5% | 85.7% |

Negative skill in five of eight windows: do not replace the production model with
Ridge. Better features, honest availability and selection/calibration validation
are higher priority than adding neural architectures. This experiment does not
resolve the existing negative-skill window or guarantee 90% coverage.
