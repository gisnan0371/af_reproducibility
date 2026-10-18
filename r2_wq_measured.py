import os
import numpy as np
import pandas as pd
import master_recompute as mr

RAW_NAME = "input_water_quality_data.csv"
STATIONS = {"Yi": {"Cobs": "LongMenDaQiao", "Cup": "TanTou"},
            "Luo": {"Cobs": "BaiMaSi", "Cup": "GaoYaZhai"}}
NUTRIENT_COLS = {"NH3": "AmmoniaNitrogen", "TN": "TotalNitrogen", "TP": "TotalPhosphorus"}
_cache = {}


def raw_path():
    for base in (getattr(mr, "WQ_FILE", None), getattr(mr, "AUX_WQ_FILE", None)):
        if base:
            p = os.path.join(os.path.dirname(str(base)), RAW_NAME)
            if os.path.isfile(p):
                return p
    for p in (RAW_NAME, os.path.join("input_data", RAW_NAME)):
        if os.path.isfile(p):
            return p
    return None


def measured_days(station, det):
    key = (station, det)
    if key not in _cache:
        p = raw_path()
        if p is None:
            raise FileNotFoundError(f"{RAW_NAME} not found")
        raw = _cache.get("__raw__")
        if raw is None:
            raw = pd.read_csv(p)
            raw["date"] = pd.to_datetime(raw[["Year", "Month", "Day"]].rename(columns=str.lower))
            raw["_st"] = raw["WaterQualityMonitoringStation"].astype(str).str.strip().str.lower()
            _cache["__raw__"] = raw
        sub = raw[(raw["_st"] == station.lower()) & raw[det].notna()]
        _cache[key] = set(sub["date"].dt.normalize())
    return _cache[key]


def build_daily_measured(river, wq_daily, rn_wide, precip, verbose=True):
    d = mr.build_daily(river, wq_daily, rn_wide, precip)
    idx = pd.to_datetime(d.index).normalize()
    masked = {}
    for prefix, station in STATIONS[river].items():
        for short, det in NUTRIENT_COLS.items():
            col = f"{prefix}_{short}"
            if col not in d.columns:
                continue
            ok = idx.isin(measured_days(station, det))
            n_before = int(d[col].notna().sum())
            # drop values that are not in the raw station file
            d.loc[~ok, col] = np.nan
            masked[col] = n_before - int(d[col].notna().sum())
    if verbose:
        tot = sum(masked.values())
        print(f"  {river}: {tot} gap-filled nutrient values set to NaN ("
              + ", ".join(f"{k} {v}" for k, v in masked.items() if v) + ")")
    return d
