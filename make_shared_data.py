#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_shared_data.py
===================
Produce the *shareable* derived datasets for the AF ("hidden amplifier") paper,
under the constraint that the raw daily discharge and diversion series are
restricted by the data provider and CANNOT be released (see README.md).

"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    import master_recompute as mr
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import master_recompute.py — put this script in the SAME folder "
        f"as master_recompute.py and run it there.\n  ({type(exc).__name__}: {exc})")

OUT   = "shared_data"
ROUND = 4

# water-quality stations to export (Yi pair + Luo longitudinal gradient, §4.5)
GRADIENT = {
    "Yi":  ["TanTou", "LongMenDaQiao"],
    "Luo": ["LingKou", "LuoNingChangShui", "QiLiPu", "GaoYaZhai", "BaiMaSi"],
}
# raw concentration column (master_recompute WQ_COL) -> shared canonical name
CONC_REN = {mr.WQ_COL["NH3"]: "NH3N", mr.WQ_COL["TN"]: "TN", mr.WQ_COL["TP"]: "TP",
            mr.WQ_COL["EC"]: "EC", mr.WQ_COL["TEMP"]: "water_temp"}


def build_event_table(river: str, d: pd.DataFrame) -> pd.DataFrame:
    """Mirror export_event_table_v2.py: events from master_recompute.label_events,
    high-first then chronological, renumbered E1..En; export only safe fields."""
    cfg = mr.RIVER[river]
    de = mr.label_events(d[d["is_irr"] & d["is_dry"]].copy(), cfg)
    ev = de[de["event_id"] >= 0].copy()
    rows = []
    for _eid, g in ev.groupby("event_id"):
        g = g.sort_index()
        has_wq = ("Cobs_NH3" in g.columns and
                  bool(pd.to_numeric(g["Cobs_NH3"], errors="coerce").notna().any()))
        rows.append({
            "reach": river, "class": g["evt"].iloc[0],
            "start_date": g.index.min().date(), "end_date": g.index.max().date(),
            "n_dry_days": int(len(g)),
            "mean_DR": round(float(g["DR"].mean()), ROUND),
            "median_AF": round(float(g["AF"].median()), ROUND),
            "has_concurrent_WQ": has_wq,
        })
    t = pd.DataFrame(rows)
    if len(t):
        t["__o"] = (t["class"] != "high").astype(int)            # high first
        t = t.sort_values(["__o", "start_date"]).drop(columns="__o").reset_index(drop=True)
        t.insert(0, "event_id", [f"E{i + 1}" for i in range(len(t))])
    else:
        t = pd.DataFrame(columns=["event_id", "reach", "class", "start_date", "end_date",
                                  "n_dry_days", "mean_DR", "median_AF", "has_concurrent_WQ"])
    return t


def build_model_frame(river: str, d: pd.DataFrame):
    """Per-day frame to reproduce the covariate-controlled NH3-N model (Table 3;
    EC tracer Table 4; season-control Table 5) on irrigation-season dry days.

    Discharge enters ONLY as standardised predictors (z-scores of log(Q+1)); the
    absolute Q_div / Q_nd are NOT recoverable from a z-score series. The
    standardisation mean/sd per reach are written separately (model_frame_scales.csv)
    so the manuscript's per-unit-log coefficient can be recovered from the per-SD
    coefficient if desired (beta_per_log_unit = beta_per_SD / sd)."""
    idx = d.index
    n = len(idx)
    col = lambda name: d[name].values if name in d.columns else np.full(n, np.nan)
    isdry = d["is_dry"].astype(bool)
    runid = (isdry != isdry.shift()).cumsum()                  # contiguous-day runs
    dry_window = np.where(isdry.values, runid.values, np.nan)  # id per dry spell
    log_qdiv = np.log(np.clip(col("Q_div"), 0, None) + 1.0)
    log_qnd  = np.log(np.clip(col("Q_nd"),  0, None) + 1.0)
    scales = {}

    def _z(arr, name):
        mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
        scales[name] = (round(mu, 6), round(sd, 6))
        return (arr - mu) / sd if sd and not np.isclose(sd, 0) else arr * 0.0

    fr = pd.DataFrame({
        "date": idx, "reach": river,
        "NH3N_down": col("Cobs_NH3"), "NH3N_up": col("Cup_NH3"),
        "z_log_Qdiv_plus1": np.round(_z(log_qdiv, f"{river}:log_Qdiv_plus1"), ROUND),
        "z_log_Qnd_plus1":  np.round(_z(log_qnd,  f"{river}:log_Qnd_plus1"),  ROUND),
        "Twater": col("Cobs_TEMP"), "API7": np.round(col("API7"), ROUND),
        "year": d["year"].values, "dry_window_id": dry_window,
        "EC_down": col("Cobs_EC"), "EC_up": col("Cup_EC"),
        "DO": col("Cobs_DO"), "turbidity": col("Cobs_TURB"),
    })
    keep = d["is_irr"].values & isdry.values
    return fr[keep].reset_index(drop=True), scales


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading via master_recompute.load_all() (raw discharge stays internal) ...")
    wq_daily, rn_wide, precip, ps = mr.load_all()

    frames, events = {}, []
    for river in ["Yi", "Luo"]:
        d = mr.build_daily(river, wq_daily, rn_wide, precip)   # authoritative daily frame
        frames[river] = d
        events.append(build_event_table(river, d))

    # (1) AF / DR daily  -- dimensionless, full period.
    #     We RECOMPUTE AF/DR explicitly from the two discharges here (audit-
    #     transparent) and assert they equal master_recompute's, so a reader can
    #     verify AF is a pure function of Q_down and Q_div. DR is forced to NA
    #     wherever AF is NA (days with Q_down <= Q_MIN are excluded), so no
    #     "AF=NA, DR=1.0" rows can occur.
    ad = []
    for river, d in frames.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            af_calc = np.where(d["Q_down"].values > mr.Q_MIN,
                               (d["Q_down"].values + d["Q_div"].values) / d["Q_down"].values,
                               np.nan)
            dr_calc = 1.0 - 1.0 / af_calc            # = Q_div/(Q_down+Q_div); NA when AF is NA
        assert np.allclose(af_calc, d["AF"].values, rtol=1e-6, atol=1e-6, equal_nan=True), \
            f"{river}: recomputed AF != master_recompute AF (pipeline mismatch)"
        t = pd.DataFrame({"date": d.index, "reach": river,
                          "AF": np.round(af_calc, ROUND), "DR": np.round(dr_calc, ROUND)})
        t.loc[t["AF"].isna(), "DR"] = np.nan         # explicit: AF NA -> DR NA
        ad.append(t)
    af = pd.concat(ad, ignore_index=True)
    af[["date", "reach", "AF", "DR"]].to_csv(f"{OUT}/af_dr_daily.csv", index=False)

    # (2) standardised abstraction index = z[ log(Q_div + 1) ]  (per reach)
    ai = []
    for river, d in frames.items():
        z = np.log(d["Q_div"].clip(lower=0) + 1.0)
        sd = z.std(ddof=0)
        z = (z - z.mean()) / sd if sd and not np.isclose(sd, 0) else z * 0.0
        ai.append(pd.DataFrame({"date": d.index, "reach": river,
                                "abstraction_index_z": np.round(z.values, ROUND)}))
    pd.concat(ai, ignore_index=True).to_csv(f"{OUT}/abstraction_index.csv", index=False)

    # (3) normalised C-Q : Qnorm = Q_down / median(Q_down) per reach (dimensionless)
    cq = []
    for river, d in frames.items():
        med = d.loc[d["Q_down"] > 0, "Q_down"].median()
        t = pd.DataFrame({"date": d.index})
        t["reach"] = river
        t["season"] = np.where(d["is_irr"].values, "irrigation", "non_irrigation")
        t["Qnorm"] = (d["Q_down"].values / med).round(ROUND)
        for k, mk in (("NH3N", "NH3"), ("TN", "TN"), ("TP", "TP")):
            c = f"Cobs_{mk}"
            if c in d.columns:
                t[k] = d[c].values
        cq.append(t)
    pd.concat(cq, ignore_index=True).to_csv(f"{OUT}/cq_normalized.csv", index=False)

    # (4) processed daily concentrations -- gradient stations (daily means), names only
    crows = []
    for river, sts in GRADIENT.items():
        for st in sts:
            s = wq_daily[wq_daily["station"] == st].copy()
            if s.empty:
                print(f"  [warn] no water-quality rows for station {st!r}")
                continue
            s.insert(0, "river", river)
            crows.append(s)
    conc = pd.concat(crows, ignore_index=True).rename(columns=CONC_REN)
    cols = ["river", "station", "date"] + [c for c in ("NH3N", "TN", "TP", "EC", "water_temp")
                                           if c in conc.columns]
    conc.sort_values(["river", "station", "date"])[cols].to_csv(
        f"{OUT}/concentrations_daily.csv", index=False)

    # (5) precipitation (covariate)
    precip[["date", "precip_mm"]].to_csv(f"{OUT}/precip_daily.csv", index=False)

    # (6) event classification
    ev_all = pd.concat(events, ignore_index=True)
    ev_all.to_csv(f"{OUT}/event_classification.csv", index=False)

    # (7) station metadata (names + role only; NO coordinates)
    meta = []
    for r, cfg in mr.RIVER.items():
        meta.append(dict(river=r, station=cfg["wq_up"], role="upstream"))
        meta.append(dict(river=r, station=cfg["wq_down"], role="downstream"))
    for r, sts in GRADIENT.items():
        for st in sts:
            if not any(m["river"] == r and m["station"] == st for m in meta):
                meta.append(dict(river=r, station=st, role="gradient"))
    pd.DataFrame(meta).to_csv(f"{OUT}/station_metadata.csv", index=False)

    # (8) model frame(s) to reproduce the covariate-controlled NH3-N model (Table 3),
    #     the EC tracer model (Table 4) and the season-control robustness (Table 5),
    #     on irrigation-season dry days. Discharge is standardised (no absolute Q).
    mf, scale_rows = [], []
    for river, d in frames.items():
        fr, sc = build_model_frame(river, d)
        mf.append(fr)
        for k, (mu, sd) in sc.items():
            r, var = k.split(":")
            scale_rows.append({"reach": r, "variable": var, "mean": mu, "sd": sd})
    pd.concat(mf, ignore_index=True).to_csv(f"{OUT}/model_frame.csv", index=False)
    # Standardisation scales are RESTRICTED: sd + the z-scores in model_frame.csv
    # would allow reconstructing absolute Q. Write them to restricted_for_review/
    # (for editors/reviewers under the data-use agreement), NOT to public shared_data/.
    os.makedirs("restricted_for_review", exist_ok=True)
    pd.DataFrame(scale_rows).to_csv("restricted_for_review/model_frame_scales.csv", index=False)

    # ---- verification (compare to results/recomputed_values.csv) ----
    print("\n=== verification (expected vs your run) ===")
    for river, d in frames.items():
        irr = d[d["is_irr"]]["AF"].dropna()
        print(f"  {river}: AF irr mean={irr.mean():.3f}  median={irr.median():.3f}  "
              f"P95={irr.quantile(0.95):.3f}   (Yi expect 2.33/1.75/6.94; Luo 2.53/2.06/5.57)")
    print(f"  events: {ev_all.groupby(['reach','class']).size().to_dict()}  "
          f"(expect Yi high=7 low=2; Luo high=15 low=0)")
    print(f"\nWrote 8 files to ./{OUT}/ (incl. model_frame.csv, {sum(len(x) for x in mf)} rows)"
          f" + restricted_for_review/model_frame_scales.csv.")
    print(f"Deposit ONLY ./{OUT}/ publicly (GitHub/Zenodo); keep restricted_for_review/"
          f" for editors/reviewers under the data-use agreement.")


if __name__ == "__main__":
    main()
