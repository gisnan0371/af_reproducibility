from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import pandas as pd
import master_recompute as mr

STA = dict(dongwan="DongWan", luhun="LuHun", luhunqu="LuHunQu", lmz="LongMenZhen",
           canals=["MaoZhuangQu", "LuHunGuangaiQu"], upper_canals=["BoYunLingQu", "YueJinQu"], tantou="TanTou")
IRR = getattr(mr, "IRR_MONTHS", [4, 5, 6, 7, 8, 9])


def col(rn, name):
    if name in rn.columns:
        return pd.to_numeric(rn[name], errors="coerce")
    print(f"  station {name} not found")
    return pd.Series(np.nan, index=rn.index)


def stats(s, label):
    s = s.dropna()
    return dict(series=label, n=len(s), mean=s.mean(), median=s.median(), pct_neg=100 * (s < 0).mean(),
                pct_pos=100 * (s > 0).mean(), sd=s.std())


def main():
    wq_daily, rn_wide, precip, ps = mr.load_all()
    rn = rn_wide.copy()
    rn.index = pd.to_datetime(rn.index)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    dw, lh, lhq, lmz = (col(rn, STA[k]) for k in ("dongwan", "luhun", "luhunqu", "lmz"))
    qdiv = sum(col(rn, c).fillna(0) for c in STA["canals"])
    irr = rn.index.month.isin(IRR)

    print("\nmonthly means (m3/s)")
    m = pd.DataFrame({"DongWan": dw, "LuHun": lh, "LuHunQu": lhq, "LongMenZhen": lmz,
                      "MaoZhuangQu": col(rn, "MaoZhuangQu"), "LuHunGuangaiQu": col(rn, "LuHunGuangaiQu")})
    print(m.groupby(m.index.month).mean().round(2).to_string())

    print("\nzero-flow share and coverage (%)")
    for name, s in m.items():
        print(f"  {name:15s} irr: zero {100*(s[irr]==0).mean():5.1f}%  cov {100*s[irr].notna().mean():5.1f}% | "
              f"non-irr: zero {100*(s[~irr]==0).mean():5.1f}%  cov {100*s[~irr].notna().mean():5.1f}%")

    print("\nten largest LongMenZhen days")
    top = m.loc[lmz.nlargest(10).index].round(1)
    print(top.to_string())
    print("\nten largest LuHunQu days")
    print(m.loc[lhq.nlargest(10).index].round(1).to_string())

    print("\nSpearman correlations, irrigation season")
    print(m[irr].corr(method="spearman").round(2).to_string())

    print("\ncore reach: dQ = Q_in - LongMenZhen - canals (m3/s)")
    hyp = {"H_legacy: Q_in = LuHunQu": lhq, "H1: Q_in = LuHun": lh, "H4: Q_in = LuHun - LuHunQu": lh - lhq.fillna(0)}
    flood = lmz > lmz.quantile(0.95)
    rows = []
    for k, qin in hyp.items():
        dq = qin - lmz - qdiv
        rows.append(stats(dq[irr & ~flood], f"{k} | irrigation, no flood"))
        rows.append(stats(dq[~irr & ~flood], f"{k} | non-irrigation, no flood"))
        rows.append(stats(dq[~flood], f"{k} | all, no flood"))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nreservoir balance")
    rows = [stats((dw - lh)[~flood], "DongWan - LuHun"),
            stats((dw - lh - lhq.fillna(0))[~flood], "DongWan - LuHun - LuHunQu")]
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nannual volumes (Mm3)")
    vol = (m * 86400 / 1e6).groupby(m.index.year).sum().round(0)
    print(vol.to_string())


if __name__ == "__main__":
    main()
