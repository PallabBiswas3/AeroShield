# AeroShield v3 final validation checklist

Use this as the merge gate for pull request #1. Record the commit SHA, commands,
dates, and outputs in the PR before marking each item complete.

## Required artifacts and provenance

- [ ] The chunked `data/surrogate_model.joblib.partNNN` and
  `data/uncertainty_models.joblib.partNNN` artifacts plus `data/model_meta.json`
  were produced by the same pipeline run. (The loader retains single-file
  compatibility for local/legacy artifacts.)
- [ ] Metadata and the uncertainty bundle contain identical feature schemas and
  provenance (seed, package versions, training-data hash, inventory hash).
- [ ] A clean checkout starts without silently dropping uncertainty intervals.
- [ ] The training command and exact dependency versions are recorded.

## Forecasting and uncertainty

- [ ] The split is chronological by timestamp: 70% train, 15% calibration,
  15% untouched test; no timestamp appears in more than one block.
- [ ] Imputation statistics and early stopping use no test data.
- [ ] Only features available at serving time are used. No contemporaneous
  co-pollutant is replaced by a constant at serving time.
- [ ] Test MAE/RMSE, persistence MAE/RMSE, skill, 90% empirical coverage, and
  mean interval width exactly match `model_meta.json`.
- [ ] Every grid and point result satisfies `0 <= lower <= median <= upper <= 500`.
- [ ] UI/API call the output a conditional estimate, not a validated future-
  horizon forecast. No 72-hour claim appears without direct horizon labels.

## Meteorology and API execution

- [ ] A valid Asia/Kolkata forecast time selects the expected Open-Meteo hour.
- [ ] An unavailable/out-of-range weather request exposes a fallback note.
- [ ] Invalid dates, wind values, and coordinates outside the Delhi grid are
  rejected with a non-2xx response.
- [ ] Backend failures return suitable 4xx/5xx status codes rather than HTTP 200
  with `{status: "error"}`.

## Plume attribution

- [ ] Fixed seeds reproduce identical ensemble output.
- [ ] A synthetic upwind source ranks above a crosswind/upwind-incompatible one.
- [ ] Relative shares sum to about 100% only across active, inventoried sources.
- [ ] Active-simulation fraction and active count are returned.
- [ ] UI and API never label relative shares as calibrated confidence or legal proof.
- [ ] Demonstration-inventory and omitted-source limitations are visible.

## Intervention and guardrails

- [ ] Ranking uses the conservative (10th-percentile) attribution share.
- [ ] Negative/non-finite exposure and PM2.5 inputs are rejected.
- [ ] Action-library efficacy is displayed as an unvalidated assumption.
- [ ] Values are called benefit proxies—not causal or guaranteed reductions.
- [ ] Low evidence leads only to monitor/verify; all actions require human approval.
- [ ] Deterministic and LLM fallbacks request inspection and never declare guilt,
  fabricate measurements, or autonomously issue a legal order.

## Release commands

- [ ] `cd backend && python -m unittest discover -s tests -v`
- [ ] `cd frontend && npm ci && npm run lint && npm run build`
- [ ] Start the API from a clean process and smoke-test `/api/model-info`,
  `/api/city-grid`, `/api/analyze-hotspot`, `/api/dispatch`, and `/api/cases`.
- [ ] Confirm no secrets, runtime cache, local database changes, or unrelated
  generated files are included in the final diff.
