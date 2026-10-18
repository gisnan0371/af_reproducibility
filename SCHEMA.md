# shared_data/ column definitions

Dates are `YYYY-MM-DD`. No absolute discharge or diversion is included: discharge enters only as
ratios (AF, DR) or as z-scores of log(Q + 1). The standardisation means and standard deviations are
kept in `restricted_for_review/` and are not part of the public files.

### af_dr_daily.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | calendar day (2018-09-01 … 2021-12-31) |
| `reach`| str  | `Yi` or `Luo` |
| `AF`   | float| Amplification Factor = (Qdown + Qdiv)/Qdown (≥ 1). Computed only where Qdown > 0.1 m³/s; otherwise `NA`. |
| `DR`   | float| diversion ratio = Qdiv/(Qdown + Qdiv) = 1 − 1/AF; `NA` wherever `AF` is `NA`. |

### abstraction_index.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | |
| `reach`| str  | `Yi` / `Luo` |
| `abstraction_index_z` | float | z-score of log(Qdiv + 1) within reach |

### precip_daily.csv
| column | type | description |
|--------|------|-------------|
| `date` | date | 2019-01-01 … 2021-12-31 |
| `precip_mm` | float | daily precipitation summed over the eight stations (mm); a dry day has < 1 mm |

### station_metadata.csv
| column | type | description |
|--------|------|-------------|
| `river`| str  | `Yi` / `Luo` |
| `station` | str | station name |
| `role` | str | `upstream` / `downstream` / `gradient` |

The four files above are unchanged from release v1.0.0, where they were written by `make_shared_data.py`.

### model_frame_r2.csv
Daily frame for the covariate-controlled models (`r2_gamm_sensitivity.R`), written by
`r2_make_model_frame.py`. Nutrient concentrations are measured values only (gap-filled values removed).

| column | type | description |
|--------|------|-------------|
| `date`, `reach` | | day and reach (`Yi` / `Luo`) |
| `year`, `month`, `yday` | int | calendar year, month and day of year |
| `dry_window_id` | float | identifier of the irrigation-season dry window (runs of ≥ 3 dry days); empty outside dry windows |
| `NH3N_down`, `NH3N_up` | float | NH₃-N at the downstream and upstream stations (mg/L) |
| `TN_down`, `TP_down` | float | TN and TP at the downstream station (mg/L) |
| `EC_down`, `EC_up` | float | specific conductance, downstream and upstream (µS/cm, 25 °C) |
| `Twater` | float | water temperature at the downstream station (°C) |
| `API7` | float | 7-day antecedent precipitation index (mm) |
| `DO`, `turbidity` | float | dissolved oxygen (mg/L) and turbidity (NTU), downstream |
| `AF`, `DR` | float | as in `af_dr_daily.csv` |
| `div_class` | str | day-level diversion class: `high`, `low` or `other` |
| `nh3_spike` | int | 1 if the daily NH₃-N mean equals the fixed value 0.50 mg/L |
| `nh3_flatrun` | int | 1 if the day lies in a run of ≥ 3 identical daily NH₃-N values |
| `nh3_clean` | int | 1 if neither of the two flags applies |
| `has_Qup` | int | 1 if the gauged upstream release is available |
| `z_log_Qdiv_plus1` | float | z-score of log(Qdiv + 1) |
| `z_log_Qnd_plus1` | float | z-score of log(Qnd + 1), Qnd = Qdown + Qdiv |
| `z_log_Qup_plus1` | float | z-score of log(Qup + 1), gauged upstream release |
| `*_lag1`, `*_lag2` | float | the same z-scores taken 1 and 2 days earlier |

Coefficients estimated from the z-scores are per standard deviation of log(Q + 1). Per-unit values,
as reported in the paper, need the restricted file `model_frame_r2_scales.csv`
(`beta_per_unit = beta_per_SD / sd`), available to editors and reviewers on request.
