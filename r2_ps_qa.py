import numpy as np
import pandas as pd

TN_Q = 0.95


def screen_ps(ps, verbose=True):
    ps = ps.copy()
    ucol = "N_unit" if "N_unit" in ps.columns else "unit"
    if ucol != "N_unit":
        ps = ps.rename(columns={ucol: "N_unit"})
    ps["date"] = pd.to_datetime(ps["date"])
    n0 = len(ps)
    num = ps.select_dtypes("number").columns.tolist()
    ps = ps.groupby(["N_unit", "date"], as_index=False)[num].mean()
    n_dedup = len(ps)
    has_conc = {"TN", "NH4N"} <= set(ps.columns)
    if not has_conc:
        if verbose:
            print("  no concentration columns in the point-source file, screen skipped")
        return ps, pd.DataFrame()
    # NH4-N cannot exceed TN on the same record
    inconsistent = ps["TN"].notna() & ps["NH4N"].notna() & (ps["NH4N"] > ps["TN"])
    tn_p95 = ps.groupby("N_unit")["TN"].quantile(TN_Q)
    ceiling = ps["N_unit"].map(tn_p95)
    implausible = ps["TN"].isna() & ps["NH4N"].notna() & (ps["NH4N"] > ceiling)
    drop = inconsistent | implausible
    rep = (ps.assign(inc=inconsistent, imp=implausible, drop=drop)
             .groupby("N_unit")
             .agg(n_days=("date", "nunique"), n_TN=("TN", "count"), n_NH4=("NH4N", "count"),
                  excluded_inconsistent=("inc", "sum"), excluded_implausible=("imp", "sum"), excluded=("drop", "sum"))
             .reset_index())
    rep["share_excluded_pct"] = (100 * rep["excluded"] / rep["n_days"]).round(1)
    rep["TN_p95_mgL"] = rep["N_unit"].map(tn_p95).round(1)
    kept = ps[~drop].reset_index(drop=True)
    if verbose:
        print(f"  point-source screen: {n0} rows, {n_dedup} outlet-days, {len(kept)} kept "
              f"({int(inconsistent.sum())} with NH4-N > TN, {int(implausible.sum())} with NH4-N > TN p95)")
        worst = rep.sort_values("share_excluded_pct", ascending=False).head(6)
        for _, r in worst.iterrows():
            if r["excluded"] > 0:
                print(f"    {r['N_unit']}: {int(r['excluded'])}/{int(r['n_days'])} days ({r['share_excluded_pct']}%)")
    return kept, rep
