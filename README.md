# Reproducibility package — "Irrigation water abstraction as a hidden amplifier of river pollution"

This package lets you (i) **regenerate** the shared derived data from the source
records, and (ii) **reproduce** the reported Amplification-Factor (AF) statistics,
the covariate-controlled models, and the figures **from the shared data alone**.


## Contents
```
af_reproducibility_package/
├── README.md                     this file
├── SCHEMA.md                     column definitions for every shared file
├── master_recompute.py           SOURCE OF TRUTH: raw records -> daily frame (auditable)
├── make_shared_data.py           runs master_recompute -> writes shared_data/*.csv (+ restricted scales)
├── reproduce_from_shared.py      AF stats, DR-ΔC, events, figures        (from shared data)
├── reproduce_model.py            Table 3/4/5 models, per-SD, Python check (from shared data)
├── reproduce_gamm_table3_4_5.R   Table 3/4/5 EXACT mgcv GAMM with AR(1)  (from shared data)
├── imputation_sensitivity.py     six daily gap-filling schemes -> SI Table S10
├── shared_data/                  the 8 PUBLIC CSVs  (see "Before you deposit")
└── restricted_for_review/        model_frame_scales.csv — RESTRICTED, not for public deposit
```

## Install

## A. Reproduce from the shared data (what a reviewer runs)
Requires only the CSVs in `shared_data/`. No raw discharge needed.
```bash
python  reproduce_from_shared.py        # AF stats, DR-ΔC descriptive correlations, event contrast, 4 figures
python  reproduce_model.py              # Table 3/4/5 models (per-SD coefficient), approximate Python check
Rscript reproduce_gamm_table3_4_5.R     # Table 3/4/5 EXACT manuscript GAMM (mgcv, AR(1) within dry window)
```
**Expected output (irrigation-season dry days):**
- AF — Yi mean/median/P95 = 2.33 / 1.75 / 6.94; Luo = 2.53 / 2.06 / 5.57.
- DR–ΔC descriptive correlation (n=201, no significance test) — Yi TN/TP/NH₃-N = 0.905 / 0.926 / 0.953; Luo = 0.951 / 0.567 / 0.635.
- Event contrast (Yi, dry days) — NH₃-N +54.0% (p=0.071); TN −10.6% (p=0.857); TP −7.1% (p=0.857).
- NH₃-N model (Table 3) / EC model (Table 4) — positive abstraction coefficient, p ≈ 0.02 / 0.003.

## Coefficient units (important)
`model_frame.csv` releases the abstraction predictor **standardised**
(`z_log_Qdiv_plus1` = z-score of log(Qdiv+1)). The fitted coefficient is therefore
**per 1 SD of log(Qdiv+1)** and is fully reproducible from the public file.
The manuscript reports the **per-unit-log** coefficients (Table 3: +0.117;
Table 4: +0.020), recovered as `beta_per_log_unit = beta_per_SD / sd`. The `sd`
lives in `restricted_for_review/model_frame_scales.csv`, which is **NOT released
publicly** (sd + the z-scores would let one reconstruct absolute discharge). If
that file is present, `reproduce_model.py` and `reproduce_gamm_table3_4_5.R` also
print the per-unit values; otherwise they print the per-SD coefficient.

## Data coverage (date ranges)
- AF / DR / normalised C–Q / abstraction index: **2018-09-01 – 2021-12-31** (daily).
- Shared daily water-quality concentrations (`concentrations_daily.csv`):
  **2020-01-01 – 2021-12-31** (downstream NH₃-N monitoring begins in 2020; rows in
  the C–Q file before 2020 are retained for date alignment but have missing
  concentrations).
- Precipitation (`precip_daily.csv`): **2019-01-01 – 2021-12-31**.
- Point sources: **2020-01-01 – 2021-12-31**.
Dry-day analyses that use water quality are unaffected by the precipitation start
date because the analysed water quality begins 2020-01-01.

## B. Regenerate the shared data from source (authors, on the machine with the raw data)
```bash
python make_shared_data.py
# -> writes 8 CSVs to shared_data/  AND  restricted_for_review/model_frame_scales.csv
```
`make_shared_data.py` imports `master_recompute.py` and recomputes AF/DR from the
two discharges with an internal `assert`, so the discharge→AF step is verifiable;
it forces `DR=NA` wherever `AF=NA` (no `AF=NA, DR=1` rows). Set the raw-file paths
at the top of `master_recompute.py` (or place the raw CSVs under `./input/`).