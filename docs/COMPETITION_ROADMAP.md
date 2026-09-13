# Competition upgrade: acceptance gates

Keep the live scenario dashboard unchanged until the new forecasting benchmark passes.

1. Forecast foundation: raw PM2.5 quality checks, completed-hour labels, exact
   t+24 joins, station-isolated lags, and purged chronological train/calibration/test.
   Historical ingestion delay is unknown; disclose optimistic hour-end availability.
2. Benchmark: history-only model first; train-only preprocessing; separate calibration
   block; station and pooled RMSE, MAE, interval coverage, width and sample counts.
   Persistence is y(t); daily seasonal persistence is identical at horizon 24h.
   Weekly seasonal persistence is y(t-144h). Compare all models on matched samples.
3. Forecast meteorology/CAMS: join only archived forecasts with valid time t+24
   and availability <= t. Never substitute target-time observations or reanalysis.
   If issue-time CAMS archives cannot be obtained, report comparison unavailable
   and begin prospective collection. Do not fabricate benchmark results.
4. Live observations: verified OpenAQ sensor metadata, units, UTC observation and
   retrieval timestamps, pagination, missing/invalid/stale flags, no silent partial
   downloads. Requires configured OpenAQ key, never a key embedded in frontend.
5. Attribution: select published inventory and compatible category/space/time scope.
   City-wide emissions shares do not establish cell-level concentration attribution.
   Report inventory agreement separately from ground-truth accuracy; retain an
   unresolved category. Top-k accuracy requires independently labelled events.
6. Vulnerability: licensed school/hospital points, dated population raster and ward
   boundaries. Avoid double counting; report coverage and spatial allocation
   assumptions. A station benchmark does not validate every 1 km grid cell.
7. Intervention: timed scripted operator task and baseline task, repeated trials,
   no claim that API latency measures actual municipal response improvement.
   External environmental expert review requires a consenting reviewer; pending.
8. Submission: architecture, benchmark report, deck, three-minute recording,
   limitations and actual deployment costs. Freeze code/data hashes in all results.

Suggested week: days 1–2 forecasting/data, day 3 archives and observation adapter,
days 4–5 attribution/vulnerability and expert review, days 6–7 freeze and recording.
Deliver a narrower validated submission if upstream data or review is unavailable.
