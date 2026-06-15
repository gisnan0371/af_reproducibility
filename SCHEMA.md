# SCHEMA — shared_data/ column definitions

All dates are `YYYY-MM-DD`. All hydrological quantities are **dimensionless**
(ratios or standardised), by design; no absolute discharge/diversion is present.
(`model_frame_scales.csv`, which carries the standardisation mean/sd, is **not** in
`shared_data/` — it is restricted; see the note at the end of this file.)

### af_dr_daily.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | calendar day (2018-09-01 … 2021-12-31) |
| `reach`| str  | `Yi` or `Luo` |
| `AF`   | float| Amplification Factor = (Qdown+Qdiv)/Qdown (≥ 1, dimensionless). Computed only where Qdown > 0.1 m³/s; otherwise `NA`. |
| `DR`   | float| diversion ratio = Qdiv/(Qdown+Qdiv) = 1 − 1/AF (0–1). **`NA` wherever `AF` is `NA`** (no `AF=NA, DR=1` rows). |

`make_shared_data.py` recomputes AF/DR from the two discharges and asserts they
equal `master_recompute.py`'s, so the discharge→AF step is auditable.

### abstraction_index.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | |
| `reach`| str  | `Yi` / `Luo` |
| `abstraction_index_z` | float | z-score of `log(Qdiv+1)` within reach (mean 0, sd 1). Standardised predictor; absolute Qdiv is NOT recoverable. |

### cq_normalized.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | |
| `reach`| str  | `Yi` / `Luo` |
| `season` | str | `irrigation` (Apr–Sep) / `non_irrigation` |
| `Qnorm`| float| downstream discharge normalised by its reach median, `Qdown/median(Qdown)` (dimensionless) |
| `NH3N`,`TN`,`TP` | float | downstream concentration (mg/L) for the C–Q plot. Rows span 2018-09-01–2021-12-31 for date alignment, but concentrations exist only from **2020-01-01**; earlier rows are blank. |

### concentrations_daily.csv
| column | type | description |
|--------|------|-------------|
| `river`| str  | `Yi` / `Luo` |
| `station` | str | monitoring-station name (see `station_metadata.csv`) |
| `date` | date | |
| `NH3N`,`TN`,`TP` | float | **daily mean** concentration (mg/L), from the preprocessed input; complete over the analysed days |
| `EC`   | float| daily mean electrical conductivity (µS/cm); may contain gaps |
| `water_temp` | float | daily mean water temperature (°C); may contain gaps |

These series cover **2020-01-01 – 2021-12-31** (731 days × 7 stations; downstream
NH₃-N monitoring begins in 2020). Each value is the mean of the available sub-daily
readings for that station-day (nutrients 4-hourly; EC/temperature hourly). No fixed minimum-reading-count is
imposed. The reported results are insensitive to the daily gap-filling method
(`imputation_sensitivity.py`; see the SI). The series is **not** synthetically
generated; it is an aggregation of the monitored sub-daily records.

### precip_daily.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | 2019-01-01 … 2021-12-31 |
| `precip_mm` | float | basin daily precipitation (mm); dry day = `< 1` mm |

### event_classification.csv
| column | type | description |
|--------|------|-------------|
| `event_id` | str | `E1`, `E2`, … (high-diversion events first, then chronological) |
| `reach` | str | `Yi` / `Luo` |
| `class` | str | `high` or `low` diversion |
| `start_date`,`end_date` | date | first/last qualifying dry day of the run |
| `n_dry_days` | int | number of qualifying dry days; intervening wet days are excluded, so the calendar span can be longer. **Event statistics use the dry days only** (filter `precip_mm < 1` within the span). |
| `mean_DR` | float | mean diversion ratio over the event |
| `median_AF` | float | median AF over the event |
| `has_concurrent_WQ` | bool | whether downstream water quality was observed during the event |

Event thresholds (daily): Yi `Qdiv≥5.0 & DR≥0.25` (high) / `Qdiv≤0.5 & DR≤0.10`
(low); Luo `Qdiv≥19 & DR≥0.45` / `Qdiv≤8 & DR≤0.20`; minimum run length 3 days.

### model_frame.csv  (for Table 3 / 4 / 5 — the covariate-controlled models)
Irrigation-season dry days only. Discharge enters as standardised predictors;
absolute Q is NOT recoverable.
| column | type | description |
|--------|------|-------------|
| `date` | date | |
| `reach`| str | `Yi` / `Luo` |
| `NH3N_down`,`NH3N_up` | float | downstream / upstream NH₃-N (mg/L) — response and control |
| `z_log_Qdiv_plus1` | float | z-score of `log(Qdiv+1)` (the abstraction predictor) |
| `z_log_Qnd_plus1` | float | z-score of `log(Qnd+1)`, Qnd = no-diversion discharge (regime control) |
| `Twater` | float | daily mean water temperature (°C) |
| `API7` | float | 7-day antecedent precipitation index (mm) |
| `year` | int | year (fixed effect) |
| `dry_window_id` | int | id of the contiguous dry spell (random-intercept grouping) |
| `EC_down`,`EC_up` | float | downstream/upstream conductivity (µS/cm) — for the EC tracer model |
| `DO`,`turbidity` | float | post-abstraction covariates (over-adjustment check only) |

### model_frame_scales.csv  (RESTRICTED — in `restricted_for_review/`, not in the public deposit)
This file is **not** part of the public `shared_data/` release. It is provided to the
editor/reviewers under the data-use agreement so the per-unit-log coefficients in
Table 3/4/5 can be verified; releasing it together with the z-scores in
`model_frame.csv` would allow reconstructing absolute Q.

| column | type | description |
|--------|------|-------------|
| `reach` | str | `Yi` / `Luo` |
| `variable` | str | `log_Qdiv_plus1` or `log_Qnd_plus1` |
| `mean`,`sd` | float | mean / sd used to standardise that variable. Recover the per-unit-log coefficient as `beta_per_log_unit = beta_per_SD / sd`. |

### station_metadata.csv
| column | type | description |
|--------|------|-------------|
| `river`| str  | `Yi` / `Luo` |
| `station` | str | station name |
| `role` | str | `upstream` / `downstream` / `gradient` |

Coordinates are deliberately omitted.
