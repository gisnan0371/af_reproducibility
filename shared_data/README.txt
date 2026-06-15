This folder must contain the 8 shared CSVs before you deposit the package.
Generate them by running, on the machine that holds the raw records:
    python ../make_shared_data.py        (writes the CSVs here)
Expected files: af_dr_daily.csv, abstraction_index.csv, cq_normalized.csv,
concentrations_daily.csv, precip_daily.csv, event_classification.csv,
station_metadata.csv, model_frame.csv, model_frame_scales.csv
See ../README.md  ("Before you deposit") and ../SCHEMA.md.
