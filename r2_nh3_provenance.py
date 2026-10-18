from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import glob
import os
import numpy as np
import pandas as pd

NH3 = "AmmoniaNitrogen"
STN = "WaterQualityMonitoringStation"
CAND_DIRS = [".", "input_data"]


def find_raw(explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    names = ["input_water_quality_data.csv", "imputed_water_quality_full.csv"]
    for d in CAND_DIRS:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    for d in CAND_DIRS:
        for p in glob.glob(os.path.join(d, "*.csv")):
            try:
                head = pd.read_csv(p, nrows=3)
            except Exception:
                continue
            if NH3 in head.columns and STN in head.columns:
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="LongMenDaQiao")
    ap.add_argument("--value", type=float, default=0.5)
    ap.add_argument("--from", dest="d0", default="2020-08-09")
    ap.add_argument("--to", dest="d1", default="2020-09-03")
    ap.add_argument("--file", default=None)
    ap.add_argument("--period", nargs=2, metavar=("START", "END"), default=None,
                    help="date range for the checks, e.g. --period 2020-01-01 2021-12-31")
    a = ap.parse_args()

    path = find_raw(a.file)
    if not path:
        print("raw water-quality file not found, use --file")
        return
    print(f"raw file: {path}")
    df = pd.read_csv(path)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

    print("\ncolumns")
    info = []
    for c in df.columns:
        s = df[c]
        u = s.dropna().unique()
        info.append(dict(column=c, dtype=str(s.dtype), n_unique=len(u),
                         sample=", ".join(map(str, u[:6]))[:60]))
    t = pd.DataFrame(info)
    print(t.to_string(index=False))
    flags = t[(t.n_unique <= 12) & (~t.column.isin(["Year", "Month", "Day", "Hour"]))]
    if len(flags):
        print("\ncolumns with few distinct values:")
        print(flags.to_string(index=False))
    else:
        print("\nno flag-like column")

    if NH3 not in df.columns or STN not in df.columns:
        print(f"\n{NH3} or {STN} missing"); return

    for c in ("Year", "Month", "Day"):
        if c not in df.columns:
            print(f"\ncolumn {c} missing"); return
    hour = df["Hour"] if "Hour" in df.columns else 0
    df["dt"] = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day,
                                   hour=hour if np.ndim(hour) else 0), errors="coerce")
    df["date"] = df["dt"].dt.normalize()
    d = df[df[STN].astype(str).str.lower() == a.station.lower()].copy()
    print(f"\nrows at {a.station}")
    key = ["dt"] + [c for c in df.columns if c not in ("dt", "date")]
    n_dup = int(d.duplicated(subset=key, keep="first").sum())
    print(f"  {len(d)} rows, {n_dup} exact duplicates ({100*n_dup/max(len(d),1):.0f}%)")
    d = d.drop_duplicates(subset=key, keep="first")
    hrs = sorted(d.loc[pd.to_numeric(d[NH3], errors="coerce").notna(), "dt"].dt.hour.unique().tolist())
    print(f"  NH3-N reading hours: {hrs}")
    if len(hrs) and len(hrs) < 24:
        print(f"  NH3-N every {24 // len(hrs)} h")
    if a.period:
        lo, hi = pd.Timestamp(a.period[0]), pd.Timestamp(a.period[1])
        d = d[(d["date"] >= lo) & (d["date"] <= hi)]
        print(f"  period {a.period[0]} to {a.period[1]} ({d['date'].nunique()} days)")
    if d.empty:
        print(f"\nno rows for {a.station}; stations: {sorted(df[STN].astype(str).unique())}")
        return
    v = pd.to_numeric(d[NH3], errors="coerce")

    print(f"\nvalue {a.value:g} in the sub-daily record")
    obs = v.dropna()
    eq = np.isclose(obs, a.value)
    below = obs[obs < a.value - 1e-9]
    print(f"  readings {len(obs)}, equal to {a.value:g}: {int(eq.sum())} ({100*eq.mean():.1f}%)")
    print(f"  below {a.value:g}: {len(below)} ({100*len(below)/max(len(obs),1):.1f}%)"
          + (f", min {below.min():.4f}" if len(below) else ""))
    print(f"  min {obs.min():.4f}, 1st percentile {obs.quantile(0.01):.4f}")
    near = obs[(obs > a.value - 0.12) & (obs < a.value + 0.12)]
    if len(near):
        print(f"\n  around {a.value:g}:")
        print(pd.cut(near, bins=np.arange(a.value - 0.12, a.value + 0.13, 0.02)).value_counts().sort_index().to_string())

    print(f"\ndays with daily mean {a.value:g}")
    daily = v.groupby(d["date"]).agg(["mean", "min", "max", "count"])
    hit = daily[np.isclose(daily["mean"], a.value)]
    if len(hit):
        uniform = np.isclose(hit["min"], hit["max"])
        print(f"  {len(hit)} days, all readings equal on {int(uniform.sum())} ({100*uniform.mean():.0f}%), "
              f"median {hit['count'].median():.0f} readings per day")
        if (~uniform).sum():
            print("  days with varying readings:")
            print(hit[~uniform].head(5).to_string())
    else:
        print("  none")

    print("\nother determinands on those days")
    others = [c for c in ("WaterTemperature", "DO", "ElectricConductance", "Turbidity",
                          "TotalNitrogen", "TotalPhosphorus", "pH", "Ph") if c in d.columns]
    if len(hit) and others:
        sub = d[d["date"].isin(hit.index)]
        rep = {}
        for c in others:
            g = pd.to_numeric(sub[c], errors="coerce").groupby(sub["date"]).agg(["min", "max", "count"])
            g = g[g["count"] > 1]
            rep[c] = 100 * np.isclose(g["min"], g["max"]).mean() if len(g) else np.nan
        print("  share of days constant through the day:")
        for c, r in rep.items():
            print(f"     {c:22s} {r:5.0f}%" if np.isfinite(r) else f"     {c:22s}   n/a")

    print(f"\n{a.d0} to {a.d1}")
    w = d[(d["date"] >= pd.Timestamp(a.d0)) & (d["date"] <= pd.Timestamp(a.d1))]
    show = ["dt", NH3] + [c for c in others if c in w.columns][:4] + \
           [c for c in flags.column.tolist() if c in w.columns][:2]
    if w.empty:
        print("  no rows")
    else:
        print(w[show].head(60).to_string(index=False))
        if len(w) > 60:
            print(f"  ... {len(w)} rows")

    os.makedirs("r2_results", exist_ok=True)
    daily.to_csv("r2_results/nh3_daily_from_raw.csv", encoding="utf-8-sig")
    print("\nsaved r2_results/nh3_daily_from_raw.csv")


if __name__ == "__main__":
    main()
