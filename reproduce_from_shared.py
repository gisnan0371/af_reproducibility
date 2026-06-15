#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_from_shared.py
========================
Reproduces:
  (A) AF summary statistics  (mean / median / P95), irrigation season
  (B) ΔC = Cobs(1 - 1/AF)  and the DR-ΔC descriptive correlation (no significance test)
  (C) event-level high- vs low-diversion contrast (Mann-Whitney U)
  (D) figures: AF time series, DR-ΔC scatter, normalised C-Q,
      Luo longitudinal gradient
  (E) (optional) the independent NH3-N response model on the standardised
      abstraction index, if `statsmodels` (or `pygam`) is installed

"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

D = "shared_data"
FIG = "figures"
IRR_MONTHS = [4, 5, 6, 7, 8, 9]
POLL = ["NH3N", "TN", "TP"]


def _load():
    af = pd.read_csv(f"{D}/af_dr_daily.csv", parse_dates=["date"])
    conc = pd.read_csv(f"{D}/concentrations_daily.csv", parse_dates=["date"])
    meta = pd.read_csv(f"{D}/station_metadata.csv")
    ev = pd.read_csv(f"{D}/event_classification.csv", parse_dates=["start_date", "end_date"])
    cq = pd.read_csv(f"{D}/cq_normalized.csv", parse_dates=["date"])
    precip = pd.read_csv(f"{D}/precip_daily.csv", parse_dates=["date"]).set_index("date")["precip_mm"]
    return af, conc, meta, ev, cq, precip


def downstream_conc(conc, meta):
    """daily downstream concentration per reach (river)."""
    out = {}
    for r in meta.river.unique():
        st = meta[(meta.river == r) & (meta.role == "downstream")].station.iloc[0]
        d = conc[conc.station == st].set_index("date")[POLL]
        out[r] = d
    return out


# --------------------------------------------------------------------------
def part_A_af_stats(af):
    print("\n(A) AF summary statistics (irrigation season)")
    rows = []
    for r, g in af.assign(month=af.date.dt.month).groupby("reach"):
        irr = g[g.month.isin(IRR_MONTHS)]["AF"].dropna()
        rows.append((r, irr.mean(), irr.median(), irr.quantile(0.95), irr.max()))
        print(f"    {r:4s}  mean={irr.mean():.3f}  median={irr.median():.3f}  "
              f"P95={irr.quantile(0.95):.3f}  max={irr.max():.3f}")
    return pd.DataFrame(rows, columns=["reach", "mean", "median", "P95", "max"])


def part_B_dr_dc(af, dconc, precip):
    print("\n(B) ΔC = Cobs(1 - 1/AF) and DR-ΔC descriptive correlation, no significance test (irrigation-season dry days, precip < 1 mm)")
    merged = {}
    for r, cd in dconc.items():
        a = af[af.reach == r].set_index("date")[["AF", "DR"]]
        m = a.join(cd, how="inner").join(precip.rename("precip_mm"))
        m = m[m.index.month.isin(IRR_MONTHS) & (m["precip_mm"].fillna(999) < 1.0)]
        for p in POLL:
            if p in m.columns:
                m[f"dC_{p}"] = m[p] * (1.0 - 1.0 / m["AF"])
        merged[r] = m
        for p in POLL:
            col = f"dC_{p}"
            if col in m.columns:
                d = m[["DR", col]].dropna()
                if len(d) > 3:
                    rr = d["DR"].corr(d[col])
                    print(f"    {r:4s} {p:5s}  descriptive r(DR,ΔC)={rr:.3f}  n={len(d)}  [no p-value; ΔC is algebraically constructed from DR]")
    return merged


def part_C_events(ev, af, dconc, precip):
    print("\n(C) event-level high- vs low-diversion contrast (downstream medians; dry days only, precip < 1 mm)")
    for r in ev.reach.unique():
        cd = dconc[r].join(precip.rename("precip_mm"))
        evr = ev[ev.reach == r]
        meds = {"high": {p: [] for p in POLL}, "low": {p: [] for p in POLL}}
        for _, e in evr.iterrows():
            seg = cd[(cd.index.date >= e.start_date.date()) & (cd.index.date <= e.end_date.date())]
            seg = seg[seg["precip_mm"].fillna(999) < 1.0]
            for p in POLL:
                if p in seg.columns and seg[p].notna().any():
                    meds[e["class"]][p].append(seg[p].median())
        for p in POLL:
            hi, lo = meds["high"][p], meds["low"][p]
            if len(hi) >= 1 and len(lo) >= 1:
                try:
                    U, pp = stats.mannwhitneyu(hi, lo, alternative="two-sided")
                except ValueError:
                    pp = np.nan
                chg = (np.median(hi) - np.median(lo)) / np.median(lo) * 100 if np.median(lo) else np.nan
                print(f"    {r:4s} {p:5s}  high(n={len(hi)}) vs low(n={len(lo)})  "
                      f"Δ={chg:+.1f}%  p={pp:.3f}")


# --------------------------------------------------------------------------
def fig_af_timeseries(af):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for r, g in af.groupby("reach"):
        ax.plot(g.date, g.AF, lw=0.7, label=r)
    ax.axhline(1, color="k", lw=0.6, ls=":")
    ax.set_ylabel("AF (–)"); ax.set_title("Amplification Factor — daily"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/AF_timeseries.png", dpi=160); plt.close(fig)


def fig_dr_dc(merged):
    fig, axes = plt.subplots(1, len(POLL), figsize=(11, 3.4), sharex=True)
    for j, p in enumerate(POLL):
        ax = axes[j]
        for r, m in merged.items():
            col = f"dC_{p}"
            if col in m.columns:
                d = m[["DR", col]].dropna()
                ax.scatter(d["DR"], d[col], s=6, alpha=0.4, label=r)
        ax.set_xlabel("DR (–)"); ax.set_title(f"ΔC {p}")
        if j == 0:
            ax.set_ylabel("ΔC (mg/L)"); ax.legend()
    fig.suptitle("Counterfactual scaling: ΔC vs diversion ratio (descriptive)")
    fig.tight_layout(); fig.savefig(f"{FIG}/DR_dC_scatter.png", dpi=160); plt.close(fig)


def fig_cq(cq):
    fig, axes = plt.subplots(1, len(POLL), figsize=(11, 3.4))
    for j, p in enumerate(POLL):
        ax = axes[j]
        if p not in cq.columns:
            continue
        for season, sub in cq.dropna(subset=[p]).groupby("season"):
            ax.scatter(sub["Qnorm"], sub[p], s=6, alpha=0.4, label=season)
        ax.set_xscale("log"); ax.set_xlabel("Q / median(Q)  (–)"); ax.set_title(f"C–Q {p}")
        if j == 0:
            ax.set_ylabel("concentration (mg/L)"); ax.legend(fontsize=7)
    fig.suptitle("Flow-matched C–Q on normalised discharge (supporting)")
    fig.tight_layout(); fig.savefig(f"{FIG}/CQ_normalized.png", dpi=160); plt.close(fig)


def fig_gradient(conc, meta):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    luo = meta[meta.river == "Luo"].station.tolist()
    means = (conc[conc.station.isin(luo)]
             .groupby("station")[["TN", "TP"]].mean().reindex(luo))
    x = np.arange(len(means))
    ax.plot(x, means["TN"], "o-", label="TN")
    ax.plot(x, means["TP"], "s-", label="TP")
    ax.set_xticks(x); ax.set_xticklabels(means.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean conc (mg/L)"); ax.set_title("Luo longitudinal gradient"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/Luo_gradient.png", dpi=160); plt.close(fig)


def part_E_optional_model(af, conc, meta):
    """Independent NH3-N response on the STANDARDISED abstraction index.
    Uses the shared abstraction_index.csv so no absolute Qdiv is needed."""
    try:
        import statsmodels.formula.api as smf
    except Exception:
        print("\n(E) skipped — install statsmodels to reproduce the covariate model")
        return
    ai = pd.read_csv(f"{D}/abstraction_index.csv", parse_dates=["date"])
    precip = pd.read_csv(f"{D}/precip_daily.csv", parse_dates=["date"]).set_index("date")["precip_mm"]
    print("\n(E) NH3-N ~ abstraction_index_z + temp + season  (Yi, dry irrigation days)")
    r = "Yi"
    st_up = meta[(meta.river == r) & (meta.role == "upstream")].station.iloc[0]
    st_dn = meta[(meta.river == r) & (meta.role == "downstream")].station.iloc[0]
    dn = conc[conc.station == st_dn].set_index("date")
    up = conc[conc.station == st_up].set_index("date")
    a = ai[ai.reach == r].set_index("date")
    df = pd.DataFrame(index=dn.index)
    df["NH3N"] = dn["NH3N"]
    df["NH3N_up"] = up["NH3N"].reindex(df.index)
    df["WT"] = dn["water_temp"] if "water_temp" in dn.columns else np.nan
    df["z"] = a["abstraction_index_z"].reindex(df.index)
    df["precip"] = precip.reindex(df.index)
    df["doy"] = df.index.dayofyear
    df = df[(df.index.month.isin(IRR_MONTHS)) & (df["precip"].fillna(0) < 1.0)]
    df = df.dropna(subset=["NH3N", "z", "WT", "NH3N_up"])
    if len(df) < 30:
        print("    too few rows after filtering; check the shared files")
        return
    m = smf.ols("np.log(NH3N+1e-3) ~ z + WT + np.sin(2*np.pi*doy/365) + "
                "np.cos(2*np.pi*doy/365) + np.log(NH3N_up+1e-3)", data=df).fit()
    print(f"    beta(z) = {m.params['z']:+.3f}   p = {m.pvalues['z']:.3f}   n = {int(m.nobs)}")
    print("    (sign/significance reproduce the manuscript; the per-log-unit β needs "
          "absolute Qdiv, which is restricted — see README)")


def main():
    os.makedirs(FIG, exist_ok=True)
    af, conc, meta, ev, cq, precip = _load()
    dconc = downstream_conc(conc, meta)
    part_A_af_stats(af)
    merged = part_B_dr_dc(af, dconc, precip)
    part_C_events(ev, af, dconc, precip)
    fig_af_timeseries(af); fig_dr_dc(merged); fig_cq(cq); fig_gradient(conc, meta)
    part_E_optional_model(af, conc, meta)
    print("\n(For the manuscript GAMM coefficients of Table 3/4/5, run reproduce_model.py,")
    print(" which fits the covariate-controlled model on shared_data/model_frame.csv.)")
    print(f"\nFigures written to ./{FIG}/")


if __name__ == "__main__":
    main()
