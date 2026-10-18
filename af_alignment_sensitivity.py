from __future__ import annotations
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import itertools
import os
import numpy as np
import pandas as pd
import master_recompute as mr

OUTDIR = "r2_results"
IRR_MONTHS = [4, 5, 6, 7, 8, 9]
Q_MIN = 0.1
LAGS = (0, 1, 2, 3)
RIVERS = {
    "Yi":  dict(canals=["MaoZhuangQu", "LuHunGuangaiQu"], q_down="LongMenZhen"),
    "Luo": dict(canals=["LuShiLuoBeiQu", "LuoNingLuoBeiQu", "YiLuoQu", "ZhongZhouQu"], q_down="BaiMaSi"),
}


def run_lengths(flag):
    return [len(list(g)) for k, g in itertools.groupby(flag) if k]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    wq_daily, rn_wide, precip, ps = mr.load_all()
    rn = rn_wide.copy()
    rn.index = pd.to_datetime(rn.index)
    rn = rn.sort_index()
    rows, steady_rows = [], []
    for river, cfg in RIVERS.items():
        canals = rn[cfg["canals"]].astype(float)
        q_down = rn[cfg["q_down"]].astype(float)
        days = rn.index[rn.index.month.isin(IRR_MONTHS) & (q_down >= Q_MIN)]
        # AF with the diversion discharge taken tau days earlier, same downstream date
        n_c = len(cfg["canals"])
        af0 = 1 + canals.sum(axis=1, min_count=n_c) / q_down
        for tau in LAGS:
            af = 1 + canals.shift(tau).sum(axis=1, min_count=n_c) / q_down
            a = af[days]
            rel = a / af0[days] - 1
            steady = pd.Series(True, index=rn.index)
            for k in range(1, tau + 1):
                steady &= ((canals.shift(k - 1) - canals.shift(k)).abs() < 1e-9).all(axis=1)
            rows.append(dict(river=river, lag_days=tau, n_days=int(a.notna().sum()),
                             AF_mean=a.mean(), AF_median=a.median(), AF_P95=a.quantile(0.95), AF_max=a.max(),
                             DR_mean=(1 - 1 / a).mean(),
                             abs_rel_diff_median_pct=100 * rel.abs().median(),
                             abs_rel_diff_P95_pct=100 * rel.abs().quantile(0.95),
                             share_days_diff_gt10pct=100 * (rel.abs() > 0.10).mean(),
                             share_days_diff_gt25pct=100 * (rel.abs() > 0.25).mean(),
                             share_days_canals_unchanged_pct=100 * steady[days].mean(),
                             r_with_lag0=a.corr(af0[days])))
            print(f"{river} lag {tau}: mean {a.mean():.2f} median {a.median():.2f} P95 {a.quantile(0.95):.2f} "
                  f"max {a.max():.1f} | days with |diff| > 10%: {100 * (rel.abs() > 0.10).mean():.0f}%, "
                  f"canals unchanged: {100 * steady[days].mean():.0f}%")
        unchanged = (canals.diff().abs() < 1e-9).loc[days]
        for c in cfg["canals"]:
            steady_rows.append(dict(river=river, canal=c,
                                    share_irr_days_unchanged_pct=100 * unchanged[c].mean(),
                                    mean_run_length_days=float(np.mean(run_lengths(unchanged[c].values)))))
        steady_rows.append(dict(river=river, canal="all canals", share_irr_days_unchanged_pct=100 * unchanged.all(axis=1).mean(),
                                mean_run_length_days=np.nan))
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/af_alignment_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(steady_rows).to_csv(f"{OUTDIR}/canal_steady_operation.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved {OUTDIR}/af_alignment_sensitivity.csv, {OUTDIR}/canal_steady_operation.csv")


if __name__ == "__main__":
    main()
