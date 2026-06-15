#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_model.py
==================
Reproduce the covariate-controlled NH3-N response model (Table 3), the EC
specific-conductance tracer model (Table 4) and the season-control robustness
(Table 5) from shared_data/model_frame.csv. 
Units / restricted data
-----------------------
"""
from __future__ import annotations
import numpy as np
import pandas as pd

D = "shared_data"
EPS = 1e-3
REQ_NH3 = ["NH3N_down", "z_log_Qdiv_plus1", "z_log_Qnd_plus1", "Twater", "API7",
           "year", "dry_window_id", "NH3N_up"]
REQ_EC  = ["EC_down", "z_log_Qdiv_plus1", "z_log_Qnd_plus1", "Twater", "API7",
           "year", "dry_window_id"]


def _sd(scales, reach, var):
    row = scales[(scales.reach == reach) & (scales.variable == var)]
    return float(row["sd"].iloc[0]) if len(row) else np.nan


def fit_one(df, response, label, sd_qdiv):
    import statsmodels.formula.api as smf
    df = df.dropna(subset=[response]).copy()
    df["y"] = np.log(df[response].clip(lower=0) + EPS)
    n = len(df)
    spline_f = ("y ~ z_log_Qdiv_plus1 + cr(z_log_Qnd_plus1, df=4) + cr(Twater, df=4) "
                "+ cr(API7, df=4) + C(year)")
    linear_f = "y ~ z_log_Qdiv_plus1 + z_log_Qnd_plus1 + Twater + API7 + C(year)"
    b = p = np.nan; how = ""
    # 1) linear mixed model: dry-window random intercept + spline smooths
    for formula, tag in [(spline_f, "spline smooths"), (linear_f, "linear covariates")]:
        try:
            m = smf.mixedlm(formula, df, groups=df["dry_window_id"].astype(str)).fit(reml=False)
            b, p = m.params["z_log_Qdiv_plus1"], m.pvalues["z_log_Qdiv_plus1"]
            how = f"mixed model (dry-window random intercept, {tag})"
            break
        except Exception:
            continue
    # 2) fallback: OLS + cluster-robust SE by dry window
    if not np.isfinite(b):
        for formula, tag in [(spline_f, "spline smooths"), (linear_f, "linear covariates")]:
            try:
                m = smf.ols(formula, df).fit(cov_type="cluster",
                                             cov_kwds={"groups": df["dry_window_id"].astype(str)})
                b, p = m.params["z_log_Qdiv_plus1"], m.pvalues["z_log_Qdiv_plus1"]
                how = f"OLS + cluster-robust SE ({tag}; mixedlm unavailable)"
                break
            except Exception:
                continue
    if not np.isfinite(b):
        print(f"\n{label}: model could not be fit on n={n} rows"); return
    per_unit = b / sd_qdiv if (sd_qdiv and np.isfinite(sd_qdiv)) else np.nan
    print(f"\n{label}  (n={n}; {how})")
    print(f"    beta[z_log_Qdiv]   = {b:+.4f}  per 1 SD of log(Qdiv+1),   p = {p:.4f}")
    if np.isfinite(per_unit):
        print(f"    beta[log(Qdiv+1)]  = {per_unit:+.4f}  per unit  (= per-SD / sd={sd_qdiv:.4f})")


def main():
    fr = pd.read_csv(f"{D}/model_frame.csv", parse_dates=["date"])
    try:
        scales = pd.read_csv("restricted_for_review/model_frame_scales.csv")
    except FileNotFoundError:
        scales = pd.DataFrame(columns=["reach", "variable", "mean", "sd"])
        print("[note] restricted_for_review/model_frame_scales.csv not present: reporting "
              "per-SD coefficients only (per-unit-log needs the restricted scales).")
    for reach in ["Yi", "Luo"]:
        sub = fr[fr.reach == reach].copy()
        sdq = _sd(scales, reach, "log_Qdiv_plus1")
        print(f"\n================  {reach} River  ================")
        d3 = sub.dropna(subset=REQ_NH3)
        if len(d3) >= 30:
            fit_one(d3, "NH3N_down", "[Table 3] NH3-N covariate-controlled model", sdq)
        else:
            print(f"  [Table 3] NH3-N: only {len(d3)} complete rows; skipped")
        d4 = sub.dropna(subset=REQ_EC)
        if len(d4) >= 30:
            fit_one(d4, "EC_down", "[Table 4] EC specific-conductance tracer model", sdq)
        else:
            print(f"  [Table 4] EC: only {len(d4)} complete rows; skipped")
    print("\nCoefficients are per 1 SD of log(Qdiv+1); divide by sd "
          "(restricted_for_review/model_frame_scales.csv) for the per-unit-log value.")
    print("Exact mgcv GAMM with AR(1): reproduce_gamm_table3_4_5.R (or the header docstring).")


if __name__ == "__main__":
    main()
