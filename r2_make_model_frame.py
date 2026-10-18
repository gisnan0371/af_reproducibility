from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import os
import numpy as np
import pandas as pd
import master_recompute as mr

from r2_wq_measured import build_daily_measured
REACHES = ["Yi", "Luo"]
OUT_PUBLIC = "shared_data"
OUT_RESTRICTED = "restricted_for_review"
ROUND = 4

QUP_STATION = {"Yi": "LuHun", "Luo": "YiYang"}
QUP_MODE = {"Yi": "station", "Luo": "station"}
QUP_MINUS = {"Yi": "LuHunQu", "Luo": None}


def event_thresholds(cfg: dict) -> dict:
    src = cfg.get("event", cfg)
    return dict(
        q_div_high=float(src.get("q_div_high", 5.0)),
        dr_high=float(src.get("dr_high", 0.25)),
        q_div_low=float(src.get("q_div_low", 0.5)),
        dr_low=float(src.get("dr_low", 0.10)),
        min_dur=int(src.get("min_dur", src.get("min_duration", 3))),
    )


def col(d: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(d[name], errors="coerce") if name in d.columns else pd.Series(np.nan, index=d.index)


def zscore(x: pd.Series, mu: float, sd: float) -> pd.Series:
    return (x - mu) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0


def censoring_flags(c: pd.Series, min_share: float = 0.05, min_run: int = 3):
    v = pd.to_numeric(c, errors="coerce")
    obs = v.dropna()
    spike_val, spike = np.nan, pd.Series(False, index=v.index)
    if len(obs):
        vc = obs.value_counts()
        if len(vc) and vc.iloc[0] / len(obs) >= min_share:
            spike_val = float(vc.index[0])
            spike = v.eq(spike_val).fillna(False)
    grp = (obs != obs.shift()).cumsum()
    runlen = obs.groupby(grp).transform("size")
    flat = pd.Series(False, index=v.index)
    flat.loc[runlen[runlen >= min_run].index] = True
    return spike_val, spike, flat


def upstream_release(river: str, d: pd.DataFrame, rn_wide) -> pd.Series:
    st = QUP_STATION.get(river)
    if rn_wide is not None and st in getattr(rn_wide, "columns", []):
        rn = rn_wide.copy(); rn.index = pd.to_datetime(rn.index)
        q = pd.to_numeric(rn[st], errors="coerce").reindex(d.index)
        if QUP_MODE.get(river) == "station_minus" and QUP_MINUS.get(river) in rn.columns:
            q = q - pd.to_numeric(rn[QUP_MINUS[river]], errors="coerce").reindex(d.index).fillna(0)
            print(f"  [{river}] Q_up = {st} - {QUP_MINUS[river]} (mode station_minus)")
        else:
            print(f"  [{river}] Q_up = {st} (mode station)")
        return q.clip(lower=0)
    print(f"  [{river}] station {st} not found, using Q_up from build_daily")
    return col(d, "Q_up")


PS_IDS = {"Yi": ["YL10001", "YL10005", "YL10016", "YL10019"],
          "Luo": ["YL10002", "YL10003", "YL10004", "YL10006", "YL10007",
                  "YL10015", "YL10017", "YL10018", "YL10012", "YL10013", "YL10014"]}
PS_LOAD_COL = "NH4N_load"
PS_START = "2020-01-01"


def ps_load_series(river: str, ps, index: pd.DatetimeIndex) -> pd.Series:
    if ps is None or not len(getattr(ps, "columns", [])):
        return pd.Series(np.nan, index=index)
    d = ps.copy()
    idcol = next((c for c in ("N_unit", "unit", "Source_ID", "ID") if c in d.columns), None)
    datecol = next((c for c in ("date", "Date") if c in d.columns), None)
    if idcol is None or datecol is None or PS_LOAD_COL not in d.columns:
        print(f"  [{river}] point-source columns not found")
        return pd.Series(np.nan, index=index)
    d[datecol] = pd.to_datetime(d[datecol])
    d = d[d[idcol].astype(str).isin(PS_IDS.get(river, []))]
    s = d.groupby(datecol)[PS_LOAD_COL].sum().reindex(index)
    s = s.fillna(0.0)
    s[index < pd.Timestamp(PS_START)] = np.nan
    print(f"  [{river}] monitored PS NH3-N load: {int(s.notna().sum())} days, mean {np.nanmean(s):.1f} kg/d")
    return s


def build_frames(river: str, wq_daily, rn_wide, precip, ps=None):
    d = build_daily_measured(river, wq_daily, rn_wide, precip).sort_index()
    thr = event_thresholds(mr.RIVER[river])
    idx = d.index

    q_div = col(d, "Q_div").clip(lower=0)
    q_down = col(d, "Q_down")
    q_nd = col(d, "Q_nd")
    if q_nd.isna().all():
        q_nd = q_down + q_div
    q_up = upstream_release(river, d, rn_wide)
    dr = col(d, "DR")
    if dr.isna().all():
        dr = q_div / q_nd.replace(0, np.nan)
    af = col(d, "AF")

    logs = {
        "log_Qdiv_plus1": np.log(q_div + 1.0),
        "log_Qnd_plus1": np.log(q_nd.clip(lower=0) + 1.0),
        "log_Qup_plus1": np.log(q_up.clip(lower=0) + 1.0),
    }
    lagged = {}
    for name, s in logs.items():
        for k in (1, 2):
            lagged[f"{name}_lag{k}"] = s.shift(k)

    is_irr = d["is_irr"].astype(bool)
    is_dry = d["is_dry"].astype(bool)
    high = is_irr & is_dry & (q_div >= thr["q_div_high"]) & (dr >= thr["dr_high"])
    low = is_irr & is_dry & (q_div <= thr["q_div_low"]) & (dr <= thr["dr_low"])
    div_class = np.where(high, "high", np.where(low, "low", "other"))

    runid = (is_dry != is_dry.shift()).cumsum()
    dry_window = np.where(is_dry.values, runid.values, np.nan)

    scales = {}
    for name, s in logs.items():
        scales[name] = (float(np.nanmean(s)), float(np.nanstd(s)))

    ps_series = ps_load_series(river, ps, d.index)
    nh3 = col(d, "Cobs_NH3")
    spike_val, nh3_spike, nh3_flat = censoring_flags(nh3)
    if np.isfinite(spike_val):
        n_obs = int(nh3.notna().sum())
        print(f"  [{river}] NH3-N = {spike_val:g} on {int(nh3_spike.sum())}/{n_obs} days "
              f"({100*nh3_spike.sum()/max(n_obs,1):.0f}%), {int(nh3_flat.sum())} days in runs of >= 3 equal values")
    else:
        print(f"  [{river}] no dominant NH3-N value")

    base = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"), "reach": river,
        "year": idx.year, "month": idx.month, "yday": idx.dayofyear,
        "dry_window_id": dry_window,
        "NH3N_down": col(d, "Cobs_NH3"), "NH3N_up": col(d, "Cup_NH3"),
        "TN_down": col(d, "Cobs_TN"), "TP_down": col(d, "Cobs_TP"),
        "EC_down": col(d, "Cobs_EC"), "EC_up": col(d, "Cup_EC"),
        "Twater": col(d, "Cobs_TEMP"), "API7": col(d, "API7"),
        "DO": col(d, "Cobs_DO"), "turbidity": col(d, "Cobs_TURB"),
        "PS_NH3_load": ps_series, "log_PS_plus1": np.log(ps_series.clip(lower=0) + 1.0),
        "AF": af, "DR": dr, "div_class": div_class,
        "nh3_spike": nh3_spike.astype(int), "nh3_flatrun": nh3_flat.astype(int),
        "nh3_clean": (~(nh3_spike | nh3_flat)).astype(int),
        "has_Qup": q_up.notna().astype(int),
    }, index=idx)

    pub = base.copy()
    res = base.copy()
    for name, s in {**logs, **lagged}.items():
        stem = name.split("_lag")[0]
        mu, sd = scales[stem]
        pub["z_" + name] = zscore(s, mu, sd).round(ROUND)
        res[name] = s.round(6)
    res["Q_div"], res["Q_down"], res["Q_up"], res["Q_nd"] = q_div, q_down, q_up, q_nd

    keep = (is_irr & is_dry).values
    pub = pub[keep].reset_index(drop=True)
    res = res[keep].reset_index(drop=True)
    return pub, res, scales, thr


def main():
    wq_daily, rn_wide, precip, ps = mr.load_all()
    try:
        from r2_ps_qa import screen_ps
        ps, _ = screen_ps(ps)
    except ImportError:
        print("  r2_ps_qa.py not found, point-source records not screened")
    os.makedirs(OUT_PUBLIC, exist_ok=True)
    os.makedirs(OUT_RESTRICTED, exist_ok=True)
    pubs, ress, scale_rows = [], [], []
    for river in REACHES:
        pub, res, scales, thr = build_frames(river, wq_daily, rn_wide, precip, ps)
        pubs.append(pub); ress.append(res)
        for k, (mu, sd) in scales.items():
            scale_rows.append({"reach": river, "variable": k, "mean": round(mu, 6), "sd": round(sd, 6)})
        n_irrdry = len(pub)
        n_fixed = int((pub["NH3N_down"].notna() & pub["Twater"].notna() & pub["EC_down"].notna()
                       & pub["API7"].notna()).sum())
        n_qup = int(pub["has_Qup"].sum())
        n_win = pub["dry_window_id"].nunique()
        n_high = int((pub["div_class"] == "high").sum()); n_low = int((pub["div_class"] == "low").sum())
        print(f"[{river}] irrigation dry days {n_irrdry}, complete cases {n_fixed}, with Q_up {n_qup}, "
              f"dry windows {n_win}, high/low days {n_high}/{n_low}")
    pd.concat(pubs).drop(columns=["PS_NH3_load", "log_PS_plus1"], errors="ignore").to_csv(
        os.path.join(OUT_PUBLIC, "model_frame_r2.csv"), index=False)
    pd.concat(ress).to_csv(os.path.join(OUT_RESTRICTED, "model_frame_r2_full.csv"), index=False)
    pd.DataFrame(scale_rows).to_csv(os.path.join(OUT_RESTRICTED, "model_frame_r2_scales.csv"), index=False)
    print(f"\nsaved {OUT_PUBLIC}/model_frame_r2.csv, {OUT_RESTRICTED}/model_frame_r2_full.csv, "
          f"{OUT_RESTRICTED}/model_frame_r2_scales.csv")


if __name__ == "__main__":
    main()
