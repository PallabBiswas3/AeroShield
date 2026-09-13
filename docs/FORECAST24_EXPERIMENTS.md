# Residual forecasting and delayed calibration experiment

Reproduce with pinned requirements, from backend:

```sh
python -m pipeline.benchmark_forecast24 --rolling --residual --adaptive
```

Residual models predict y(t+24)-y(t). Adaptive intervals use the most recent
168 matured conformity scores from the same station, with a minimum of 50.
Scores enter only when target_time < issue_time, including a 24h feedback delay.
Insufficient station history falls back to the original pooled calibration.
This is retrospective simulated feedback assuming hour-end availability; ingestion
delay must still be measured. No new operational forecast has been deployed.

| Window | Original skill | Residual skill | Original coverage | Residual + adaptive coverage | New mean width µg/m³ |
|---|---:|---:|---:|---:|---:|
| Apr 29 | 16.6% | 11.5% | 83.1% | 91.8% | 103.59 |
| May 27 | 7.6% | 10.4% | 96.7% | 91.6% | 73.82 |
| Jun 24 | 11.2% | 8.9% | 82.8% | 90.2% | 84.89 |
| Jul 22 | 11.2% | 13.6% | 94.6% | 90.0% | 49.56 |
| Aug 19 | 11.3% | 11.9% | 90.7% | 93.5% | 45.67 |
| Sep 16 | -63.5% | -44.5% | 94.9% | 91.2% | 17.37 |
| Oct 14 | 18.9% | 14.5% | 66.8% | 87.2% | 10.36 |
| Nov 11 | 10.2% | 13.9% | 85.7% | 86.7% | 14.36 |

Conclusion: calibration improves substantially, with potentially costly interval
width; residual modelling is not uniformly better and still loses to persistence
in September. Do not present this as a solved benchmark. These reused development
windows cannot serve as independent confirmation after model selection.

Next: train-period-only model selection between persistence, ridge regression and
boosting, archived forecast meteorology, independent prospective validation and
station-wise width/coverage checks. Deep sequence models are not yet justified by
the sparse five-station history.

Additional experiment: `--gate` selects persistence per station if the residual
model loses on the first half of the pre-test calibration period (or fewer than
50 eligible scores exist). Targets crossing the midpoint are purged; only the
second half calibrates intervals. This reduces September's loss to -24.3%, but
also discards April's model gain (skill becomes 0%). September coverage is 89.9%;
April coverage is 90.7%. Other reported windows retain the residual/adaptive values.
Selection is not a guarantee of future skill. Retain all candidate results;
do not promote a method based on one improved window.

Wind audit: hosted forecast request returned HTTP 429 on 2026-09-13. The displayed
5m/s and 120° were manual fallback values. A separate direct provider request
succeeded, so this is deployment/provider availability, not proof the API variable
is unsupported. Cooldown now prevents repeated retries, UI prominently labels
scenario wind, arrow points downwind, and initial time is Delhi-local.
