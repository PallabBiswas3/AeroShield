# AeroShield one-week high-impact demo plan

## Demo promise

A city officer can move from a pollution signal to a documented, human-approved
verification action while seeing uncertainty, live-data freshness, satellite
context, affected groups, and bilingual public guidance.

## Implemented in the first slice

- resilient live/coarse air-quality context with memory and disk-cache modes;
- a clearly labelled external 24-hour Open-Meteo/CAMS baseline;
- NASA FIRMS VIIRS adapter with no fabricated fallback detections;
- wind-compatible thermal-anomaly screening;
- deterministic English/Hindi health templates driven by the interval upper bound;
- signal-to-recommendation processing timestamps; and
- dashboard controls and evidence cards for these capabilities.

## Remaining critical path

1. Train a direct AeroShield `t+24 h` target using only time-`t` observations and
   forecast-available covariates.
2. Report untouched chronological MAE/RMSE against 24-hour persistence and a
   seasonal baseline; report conformal coverage and width.
3. Add a recent station observation adapter with timestamp, station distance,
   cache state, and a committed demo snapshot whose provenance is documented.
4. Add verified school/hospital GeoJSON and counts to the selected-grid workflow.
5. Obtain the FIRMS key and capture a valid cache before the recorded demo.
6. Ask at least one environmental/domain reviewer to score recommendation
   relevance, feasibility, and evidence quality using a fixed rubric.
7. Measure only system processing time unless a real manual-workflow comparison
   is conducted; never present an invented agency response-time reduction.

## Submission gates

- no future-horizon claim without a direct target and baseline table;
- no satellite marker without provider timestamp and source metadata;
- no source share labelled confidence, probability, or proof;
- no health message presented as individual medical advice;
- live API failure must leave the cached demo usable and visibly stale;
- the presentation and video must use the same metrics committed in metadata.
