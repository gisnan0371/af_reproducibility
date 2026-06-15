#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imputation_sensitivity.py
=========================
Sensitivity of the Amplification-Factor (AF) results to DAILY-level gap-filling
(imputation) of the water-quality (WQ) series.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    import master_recompute as mr
except Exception as e:                                              # pragma: no cover
    sys.exit("Could not import master_recompute.py -- put this script in the SAME "
             f"folder as master_recompute.py. ({e})")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
METHODS = ["none", "linear", "time", "ffill_bfill", "nearest", "spline"]
MAX_GAP = 3        # fill only INTERIOR gaps up to this many consecutive days.
                   # set large (e.g. 999) to fill all interior gaps; tune to mimic
                   # your real preprocessing if you know it.
RIVERS  = ["Yi", "Luo"]
OUTDIR  = getattr(mr, "RESULTS_DIR", Path("./results"))
Path(OUTDIR).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
def impute_wq_daily(wq_daily, method, max_gap):
    """Return wq_daily (long: station,date,<determinands>) with each station's daily
    series reindexed to the full daily range and gap-filled by `method`.
    method='none' returns the frame unchanged (no fill = current pipeline)."""
    if method == "none":
        return wq_daily.copy()
    full = pd.date_range(mr.FULL_START, mr.FULL_END, freq="D")
    val_cols = [c for c in wq_daily.columns if c not in ("station", "date")]
    out = []
    for stn, g in wq_daily.groupby("station"):
        g = g.set_index("date").sort_index().reindex(full)
        for c in val_cols:
            s = g[c]
            if s.notna().sum() < 2:                # nothing to interpolate from
                continue
            try:
                if method == "linear":
                    g[c] = s.interpolate(method="linear", limit=max_gap, limit_area="inside")
                elif method == "time":
                    g[c] = s.interpolate(method="time", limit=max_gap, limit_area="inside")
                elif method == "nearest":
                    g[c] = s.interpolate(method="nearest", limit=max_gap, limit_area="inside")
                elif method == "ffill_bfill":
                    g[c] = s.ffill(limit=max_gap).bfill(limit=max_gap)
                elif method == "spline":
                    g[c] = s.interpolate(method="spline", order=2, limit=max_gap, limit_area="inside")
                else:
                    raise ValueError(f"unknown method {method}")
            except Exception:
                pass                               # leave column unfilled if a method fails here
        g["station"] = stn
        g.index.name = "date"
        out.append(g.reset_index())
    res = pd.concat(out, ignore_index=True)
    return res[["station", "date"] + val_cols]


def metrics_for(d, cfg):
    """Concentration-dependent + AF metrics, mirroring master_recompute definitions."""
    m = {}
    irr = d[d["is_irr"]]
    af = irr["AF"].dropna()
    m["AF_irr_mean"]   = round(float(af.mean()), 3)   if len(af) else np.nan
    m["AF_irr_median"] = round(float(af.median()), 3) if len(af) else np.nan
    m["AF_irr_P95"]    = round(float(af.quantile(.95)), 3) if len(af) else np.nan

    dry = d[d["is_irr"] & d["is_dry"]]
    for p in mr.POLL:
        cobs = dry.get(f"Cobs_{p}")
        dct  = dry.get(f"dCtot_{p}")
        m[f"n_dryCobs_{p}"] = int(cobs.notna().sum()) if cobs is not None else 0
        m[f"dCtot_med_{p}"] = (round(float(dct.dropna().median()), 4)
                               if (dct is not None and dct.notna().any()) else np.nan)
        if dct is not None:                          # dose-response: Pearson DR vs dCtot
            x, y = dry["DR"], dct
            mm = x.notna() & y.notna()
            m[f"DRdCtot_r_{p}"] = (round(float(stats.pearsonr(x[mm], y[mm])[0]), 3)
                                   if mm.sum() > 3 else np.nan)

    # event contrast (mirror events(): label -> event medians -> high vs low Mann-Whitney)
    de  = mr.label_events(dry.copy(), cfg)
    ev  = de[de["event_id"] >= 0]
    med = ev.groupby(["evt", "event_id"]).median(numeric_only=True).reset_index()
    hi  = med[med["evt"] == "high"]; lo = med[med["evt"] == "low"]
    m["n_high"] = int(len(hi)); m["n_low"] = int(len(lo))
    for p in mr.POLL:
        c = f"Cobs_{p}"
        if c in hi.columns and c in lo.columns and len(hi) and len(lo):
            a, b = hi[c].dropna(), lo[c].dropna()
            if len(a) and len(b):
                _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
                chg = (a.median() - b.median()) / b.median() * 100 if b.median() else np.nan
                m[f"hiVlo_chg%_{p}"] = round(float(chg), 1)
                m[f"hiVlo_p_{p}"]    = round(float(pu), 3)
    return m


def _fence(df):                                      # markdown without needing 'tabulate'
    return "```\n" + df.to_string() + "\n```\n\n"


def run_comparison(tag=""):
    print(f"Loading via master_recompute.load_all() {tag}...")
    wq0, rn, precip, ps = mr.load_all()
    print(f"  baseline wq_daily rows={len(wq0)}; stations={sorted(wq0['station'].unique())}")
    print(f"  methods={METHODS} | MAX_GAP={MAX_GAP} d (interior gaps only)\n")

    rows = []
    for method in METHODS:
        wq = impute_wq_daily(wq0, method, MAX_GAP)
        for river in RIVERS:
            d = mr.build_daily(river, wq, rn, precip)
            for k, v in metrics_for(d, mr.RIVER[river]).items():
                rows.append({"method": method, "river": river, "metric": k, "value": v})
    tidy = pd.DataFrame(rows)
    sfx = ("_" + tag.strip("()").replace(" ", "_")) if tag else ""
    tidy.to_csv(Path(OUTDIR) / f"imputation_comparison_tidy{sfx}.csv", index=False)

    metric_order = tidy.drop_duplicates("metric")["metric"].tolist()
    md = [f"# Imputation sensitivity {tag}".rstrip(),
          f"\n`none` = current pipeline (no daily gap-filling) = published values. "
          f"MAX_GAP={MAX_GAP} d, interior gaps only.\n"]
    for river in RIVERS:
        sub = tidy[tidy["river"] == river]
        piv = (sub.pivot(index="metric", columns="method", values="value")
                  .reindex(index=metric_order,
                           columns=[c for c in METHODS if c in sub["method"].unique()]))
        print(f"\n================  {river} River  ================")
        print(piv.to_string())
        md += [f"## {river} River\n", _fence(piv)]

        dev_rows = []
        for met in piv.index:
            base = piv.loc[met, "none"] if "none" in piv.columns else np.nan
            if pd.isna(base):
                continue
            diffs = [piv.loc[met, c] - base for c in piv.columns
                     if c != "none" and pd.notna(piv.loc[met, c])]
            maxdev = max((abs(x) for x in diffs), default=np.nan)
            pct = (maxdev / abs(base) * 100) if (base not in (0,) and pd.notna(maxdev)) else np.nan
            dev_rows.append({"metric": met, "none": base,
                             "max_abs_dev": round(maxdev, 4) if pd.notna(maxdev) else np.nan,
                             "max_%dev_vs_none": round(pct, 2) if pd.notna(pct) else np.nan})
        dev = pd.DataFrame(dev_rows)
        print(f"\n--- {river}: deviation from no-imputation ---")
        print(dev.to_string(index=False))
        md += [f"### {river}: deviation from `none`\n", _fence(dev)]

        conc = dev[~dev["metric"].str.startswith("AF_")]
        if conc["max_%dev_vs_none"].notna().any():
            worst = conc.loc[conc["max_%dev_vs_none"].astype(float).idxmax()]
            verdict = (f"- **{river}**: AF stats identical across methods (discharge-only). "
                       f"Largest concentration-side change vs no-imputation: "
                       f"{worst['metric']} = {worst['max_%dev_vs_none']}%.")
        else:
            verdict = f"- **{river}**: AF stats identical; no concentration metrics available."
        print("\n" + verdict)
        md += ["## Headline\n", verdict + "\n"]

    (tidy.pivot_table(index=["river", "metric"], columns="method", values="value")
         .to_csv(Path(OUTDIR) / f"imputation_comparison_pivot{sfx}.csv"))
    with open(Path(OUTDIR) / f"imputation_comparison_summary{sfx}.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"\nWrote: {OUTDIR}/imputation_comparison_tidy{sfx}.csv, _pivot{sfx}.csv, _summary{sfx}.md")
    return tidy


def main():
    run_comparison()


if __name__ == "__main__":
    main()

# ======================================================================
# OPTIONAL -- compare TWO input files end-to-end (e.g. a truly RAW,
# never-interpolated WQ file vs the imputed one). This most directly answers
# "did the prior interpolation of the original data matter", IF you have the
# pre-interpolation file. Uncomment, set paths, and run again:
#
#   mr.WQ_FILE     = r"D:\...\RAW_water_quality_data.csv"     # raw nutrients
#   mr.AUX_WQ_FILE = r"D:\...\RAW_water_quality_full.csv"     # raw T/DO/EC
#   mr.INPUT_DIR   = Path(mr.WQ_FILE).parent
#   run_comparison(tag="(RAW input)")
# ======================================================================
