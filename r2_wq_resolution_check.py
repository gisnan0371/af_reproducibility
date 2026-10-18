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
OUTDIR = "r2_results"
VARS = {"NH3-N": "Cobs_NH3", "TN": "Cobs_TN", "TP": "Cobs_TP", "EC": "Cobs_EC"}
IRR = getattr(mr, "IRR_MONTHS", [4, 5, 6, 7, 8, 9])


def thresholds(cfg):
    src = cfg.get("event", cfg)
    return (float(src.get("q_div_high", 5.0)), float(src.get("dr_high", 0.25)),
            float(src.get("q_div_low", 0.5)), float(src.get("dr_low", 0.10)))


def runs_of_equal(s: pd.Series):
    v = s.dropna()
    if v.empty:
        return []
    grp = (v != v.shift()).cumsum()
    out = []
    for _, g in v.groupby(grp):
        if len(g) >= 3:
            out.append((len(g), float(g.iloc[0]), g.index.min().date(), g.index.max().date()))
    return sorted(out, reverse=True)[:8]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wq_daily, rn_wide, precip, ps = mr.load_all()
    pd.set_option("display.width", 200)
    rows = []
    for river in ("Yi", "Luo"):
        d = build_daily_measured(river, wq_daily, rn_wide, precip).sort_index()
        qh, dh, ql, dl = thresholds(mr.RIVER[river])
        qdiv = pd.to_numeric(d["Q_div"], errors="coerce")
        dr = pd.to_numeric(d["DR"], errors="coerce")
        ok = d["is_irr"].astype(bool) & d["is_dry"].astype(bool)
        hi = ok & (qdiv >= qh) & (dr >= dh)
        lo = ok & (qdiv <= ql) & (dr <= dl)
        print(f"\n{river}: {int(ok.sum())} irrigation dry days ({int(hi.sum())} high, {int(lo.sum())} low)")
        for name, ccol in VARS.items():
            if ccol not in d.columns:
                continue
            s = pd.to_numeric(d[ccol], errors="coerce")[ok]
            s = s.dropna()
            if s.empty:
                continue
            vc = s.value_counts().head(6)
            uniq = np.sort(s.unique())
            step = np.min(np.diff(uniq)) if len(uniq) > 1 else np.nan
            print(f"\n  {name}: n = {len(s)}, {s.nunique()} distinct values, smallest step {step:.4g}")
            print(f"     most frequent: " + ", ".join(f"{v:g} x{c} ({100*c/len(s):.0f}%)" for v, c in vc.items()))
            top_v, top_c = float(vc.index[0]), int(vc.iloc[0])
            sh_hi = float(((pd.to_numeric(d[ccol], errors="coerce") == top_v) & hi).sum())
            sh_lo = float(((pd.to_numeric(d[ccol], errors="coerce") == top_v) & lo).sum())
            n_hi_obs = float((pd.to_numeric(d[ccol], errors="coerce").notna() & hi).sum())
            n_lo_obs = float((pd.to_numeric(d[ccol], errors="coerce").notna() & lo).sum())
            print(f"     {top_v:g}: {100*top_c/len(s):.0f}% of days; high {sh_hi:.0f}/{n_hi_obs:.0f}, "
                  f"low {sh_lo:.0f}/{n_lo_obs:.0f}")
            rr = runs_of_equal(pd.to_numeric(d[ccol], errors="coerce")[ok])
            if rr:
                print("     longest runs of equal values:")
                for L, v, a, b in rr[:5]:
                    print(f"       {L:3d} days at {v:g}   {a} -> {b}")
            else:
                print("     no run of 3+ equal values")
            rows.append(dict(reach=river, variable=name, n=len(s), n_distinct=s.nunique(),
                             min_gap=step, top_value=top_v, top_share_pct=100 * top_c / len(s),
                             top_share_high_pct=100 * sh_hi / max(n_hi_obs, 1),
                             top_share_low_pct=100 * sh_lo / max(n_lo_obs, 1),
                             longest_equal_run=rr[0][0] if rr else 0))
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/wq_resolution_check.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved {OUTDIR}/wq_resolution_check.csv")


if __name__ == "__main__":
    main()
