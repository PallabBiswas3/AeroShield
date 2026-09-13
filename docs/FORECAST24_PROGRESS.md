# Day-ahead development benchmark

Reproduce from backend with pinned requirements:

```sh
python -m unittest discover -s tests -v
python -m pipeline.benchmark_forecast24
python -m pipeline.benchmark_forecast24 --rolling
```

Raw data SHA256: 68ab9df7118a2a71344bad0ce40b69be6a2500859a2e22ecb11436df8d27181d

These are retrospective history-only station forecasts, not validated 1km outputs.
567 of 55,631 PM2.5 rows fail basic value/time/unit checks. Additional sensor
quality assurance and source provenance verification remain necessary.
Historical ingestion timestamps are missing; hour-end availability is assumed.

## Rolling development results

Fixed LightGBM hyperparameters, 90-day initial history, 28-day calibration and
28-day test issue windows; targets are purged at train/calibration boundaries.
Calibration targets are observed before each test window. No tuning to test labels.

| Test start (UTC) | Stations | Samples | RMSE skill vs persistence | Nominal 90% coverage |
|---|---:|---:|---:|---:|
| 2025-04-29 | 5 | 2881 | 16.6% | 83.1% |
| 2025-05-27 | 5 | 2850 | 7.6% | 96.7% |
| 2025-06-24 | 5 | 2911 | 11.2% | 82.8% |
| 2025-07-22 | 5 | 2575 | 11.2% | 94.6% |
| 2025-08-19 | 4 | 2234 | 11.3% | 90.7% |
| 2025-09-16 | 2 | 673 | -63.5% | 94.9% |
| 2025-10-14 | 1 | 603 | 18.9% | 66.8% |
| 2025-11-11 | 1 | 670 | 10.2% | 85.7% |

Negative skill means worse than persistence. Full commands report station-specific
RMSE, MAE, coverage, width and matched weekly-baseline metrics.
Late-year station attrition prevents a city-wide winter conclusion.
These windows are now development evidence: reserve new prospective data for final validation.

## Next acceptance gates

1. Verify raw sensor identities/units and obtain multi-station winter observations.
2. Investigate September error and October undercoverage without retuning on these tests.
3. Add issued weather forecasts with explicit availability timestamps; archive CAMS
   prospectively if a suitable historical issue-time archive is unavailable.
4. Evaluate recent-window/station calibration with minimum sample requirements,
   reporting empirical coverage rather than claiming a time-series guarantee.
5. Add model/feature artifacts and inference only after reproducible evaluation.

## Remaining programme

Observation ingestion: verified sensor metadata, request status, pagination and
freshness flags. Vulnerability: licensed ward boundaries, schools/hospitals and
population allocation with dates and coverage. Attribution: independently sourced
inventory agreement, unresolved sources, and no invented event labels. Intervention:
repeated operator timing and a consenting domain expert's review. Submission:
architecture, evaluation report, deck and demo recording tied to a frozen revision.

No expert review, CAMS benchmark, vulnerability layer or live t+24 deployment has
been completed by this checkpoint. Network measurements include client routing;
prior latency observations do not isolate Render cold-start time.
