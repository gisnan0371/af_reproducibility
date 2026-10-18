from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import os
import numpy as np
import pandas as pd
import master_recompute as mr

OUTDIR = "r2_results"
IRR = getattr(mr, "IRR_MONTHS", [4, 5, 6, 7, 8, 9])
PRECIP_DRY = float(getattr(mr, "PRECIP_DRY", getattr(mr, "PRECIP_THR", 1.0)))


def col(rn, name):
    return pd.to_numeric(rn[name], errors="coerce") if name in rn.columns else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mean", type=float, default=-14.5)
    ap.add_argument("--target-negpct", type=float, default=88.0)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    wq_daily, rn_wide, precip, ps = mr.load_all()
    rn = rn_wide.copy(); rn.index = pd.to_datetime(rn.index)
    lhq, lh, lmz = col(rn, "LuHunQu"), col(rn, "LuHun"), col(rn, "LongMenZhen")
    mz, lg = col(rn, "MaoZhuangQu"), col(rn, "LuHunGuangaiQu")
    if any(x is None for x in (lhq, lh, lmz, mz, lg)):
        print("missing station column"); return
    qdiv = mz.fillna(0) + lg.fillna(0)

    p_s = None
    try:
        d = mr.build_daily("Yi", wq_daily, rn_wide, precip)
        for c in ("precip", "precip_mm", "precip_sum", "P"):
            if c in d.columns:
                p_s = pd.to_numeric(d[c], errors="coerce").reindex(rn.index); break
    except Exception:
        pass

    FORMULAS = {
        "A legacy   : LuHunQu - LongMenZhen - Qdiv": lhq - lmz - qdiv,
        "B corrected: LuHun   - LongMenZhen - Qdiv": lh - lmz - qdiv,
        "C corrected: LuHunQu - LongMenZhen":        lhq - lmz,
        "D no Qdiv  : LuHun   - LongMenZhen":        lh - lmz,
        "E legacy-MZ: LuHunQu - LongMenZhen - MaoZhuangQu":    lhq - lmz - mz.fillna(0),
        "F legacy-LG: LuHunQu - LongMenZhen - LuHunGuangaiQu": lhq - lmz - lg.fillna(0),
    }
    idx = rn.index
    q95, q98, q99 = [lmz.quantile(q) for q in (0.95, 0.98, 0.99)]
    FILTERS = {
        "all days with data": pd.Series(True, index=idx),
        "exclude LongMenZhen top 5%": lmz <= q95,
        "exclude LongMenZhen top 2%": lmz <= q98,
        "exclude LongMenZhen top 1%": lmz <= q99,
        "exclude LongMenZhen > 100 m3/s": lmz <= 100,
        "exclude LongMenZhen > 200 m3/s": lmz <= 200,
        "exclude Jul-Sep 2021 flood": pd.Series(~((idx >= "2021-07-01") & (idx <= "2021-09-30")), index=idx),
        "irrigation season only": pd.Series(idx.month.isin(IRR), index=idx),
        "non-irrigation only": pd.Series(~idx.month.isin(IRR), index=idx),
        "2019 onwards": pd.Series(idx.year >= 2019, index=idx),
        "all five stations reported": lhq.notna() & lh.notna() & lmz.notna() & mz.notna() & lg.notna(),
    }
    if p_s is not None:
        FILTERS["dry days only (precip < %.1f mm)" % PRECIP_DRY] = p_s < PRECIP_DRY
        FILTERS["irrigation AND dry"] = pd.Series(idx.month.isin(IRR), index=idx) & (p_s < PRECIP_DRY)
    else:
        print("  no precipitation column in build_daily, dry-day filters skipped")

    rows = []
    for fname, series in FORMULAS.items():
        for gname, mask in FILTERS.items():
            m = pd.Series(np.asarray(mask), index=idx) if not isinstance(mask, pd.Series) else mask
            v = series[m.reindex(idx).fillna(False).astype(bool)].dropna()
            if len(v) < 30:
                continue
            rows.append(dict(formula=fname, day_filter=gname, n=len(v),
                             mean=round(float(v.mean()), 2), median=round(float(v.median()), 2),
                             pct_negative=round(100 * float((v < 0).mean()), 1)))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUTDIR}/yi_s4_filter_scan.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)

    leg = t[t.formula.str.startswith("A")].copy()
    leg["dist"] = (leg["mean"] - a.target_mean).abs() / abs(a.target_mean) \
                  + (leg["pct_negative"] - a.target_negpct).abs() / a.target_negpct
    leg = leg.sort_values("dist")
    print(f"\nlegacy formula, filters closest to mean {a.target_mean} / {a.target_negpct}% negative")
    print(leg.head(8).drop(columns="dist").to_string(index=False))
    best = leg.iloc[0]
    ok = abs(best["mean"] - a.target_mean) < 0.6 and abs(best["pct_negative"] - a.target_negpct) < 4
    print(f"best: {best['day_filter']} -> mean {best['mean']}, {best['pct_negative']}% negative"
          + ("" if ok else " (no close match)"))

    print("\ncorrected residual by filter")
    cor = t[t.formula.str.startswith(("B", "C"))]
    print(cor.pivot_table(index="day_filter", columns="formula", values=["n", "mean", "pct_negative"]).to_string())

    if ok:
        r = t[(t.formula.str.startswith("B")) & (t.day_filter == best["day_filter"])]
        if len(r):
            r = r.iloc[0]
            print(f"\ncorrected, {best['day_filter']} (n = {int(r['n'])}): "
                  f"mean {r['mean']:.1f} m3/s, {r['pct_negative']:.0f}% negative")
    else:
        print("\nno filter matches; legacy formula with the no-data days counted as zero:")
        leg_all = (lhq - lmz - qdiv).dropna()
        n_idx = len(idx)
        diluted_mean = leg_all.mean() * len(leg_all) / n_idx
        diluted_neg = 100 * (leg_all < 0).sum() / n_idx
        print(f"  {len(leg_all)} days with data: mean {leg_all.mean():.2f}, "
              f"{100*(leg_all<0).mean():.1f}% negative")
        print(f"  full {n_idx}-day index, {n_idx-len(leg_all)} days as zero: "
              f"mean {diluted_mean:.2f}, {diluted_neg:.1f}% negative")
    print(f"\nsaved {OUTDIR}/yi_s4_filter_scan.csv")


if __name__ == "__main__":
    main()
