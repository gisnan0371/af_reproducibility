# Irrigation abstraction and dilution-capacity loss: analysis code

Scripts for "Irrigation water abstraction as a hidden amplifier of river pollution: observation-based
empirical evidence from a dual-river system in central China" (Journal of Hydrology).

## Requirements

Python 3.9 or later with pandas, numpy, scipy and matplotlib; R 4.2 or later with mgcv and nlme for
`r2_gamm_sensitivity.R`.

## Files

The scripts sit in one folder together with `master_recompute.py`, which reads the monitoring
records and builds the daily series.

| Script | Output |
|---|---|
| `r2_station_topology_check.py` | check that LuHunQu equals the LuHun release minus the two Yi offtakes |
| `r2_flow_audit_fix.py` | derived-series check and segment flow balance |
| `r2_match_yi_s4.py` | Yi core-reach residual under alternative day filters |
| `r2_wq_measured.py` | removes gap-filled nutrient values (imported by other scripts) |
| `r2_ps_qa.py` | point-source record screen (imported by other scripts) |
| `r2_make_model_frame.py` | model frame in `shared_data/` and `restricted_for_review/` |
| `r2_measured_coverage.py` | measured EC and water temperature, and point-source records, on the model days |
| `r2_events_audit.py` | diversion events, event contrasts (`event_contrasts_all.csv`) and the definition grid |
| `r2_wq_resolution_check.py` | NH3-N fixed-value check |
| `r2_nh3_provenance.py` | sub-daily NH3-N record at LongMenDaQiao |
| `af_alignment_sensitivity.py` | AF recomputed with diversion discharge lagged 0–3 d; per-canal steady-operation shares |
| `r2_gamm_sensitivity.R` | covariate-controlled models (mgcv::bam), including `r2_all_fits.csv` |
| `generate_all_figures_v5_9.py` | Figs. 2-4 and 6-12, Figs. S1-S9, and the tables |

Fig. 1 (study area) and Fig. 5 (conceptual diagram) were drawn separately and are not produced here.

## Running

```
python run_all.py
python run_all.py --boot 200
python run_all.py --only 1 3 6
```

`--topology` adds the station-topology check. The R step needs Rscript on the PATH or in the active
conda environment. Logs are written to `r2_log_step*.txt`, results to `r2_results/`, and figures and
tables to `output/`. Input data are read from `input_data/` in the same folder; set `AF_BASE_DIR` to use
another location.

## Input data

The raw monitoring records are not public; they are available from the authors on reasonable request.
To run the scripts, put the raw files in a folder `input_data/` next to the scripts and set the file
paths at the top of `master_recompute.py` (`WQ_FILE`, `AUX_WQ_FILE`, `RUNOFF_FILE`, `METRO_FILE`,
`PS_FILE`). Set `AF_BASE_DIR` if the data folder is elsewhere.

## Shared data

`shared_data/` holds the public derived data (column definitions in `SCHEMA.md`): daily AF and DR,
the standardised abstraction index, daily precipitation, station roles, and `model_frame_r2.csv`, the
daily frame used by the covariate-controlled models. Discharge appears only as ratios or z-scores.
The standardisation constants and the unscaled frame (`restricted_for_review/`) are available to
editors and reviewers on request.

## Versions

- v1.2.0: analysis scripts for the revised paper, including the AF date-alignment sensitivity.
- v1.0.0: first deposit (`make_shared_data.py`, `reproduce_*.py`); kept for reference, superseded by v1.2.0.
