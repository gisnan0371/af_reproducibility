from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import itertools
import numpy as np
import pandas as pd
import master_recompute as mr

OUTDIR = "r2_results"
TOL = 0.11
CHANNEL = {
    "Yi": ["LuanChuan", "TanTou", "DongWan", "LuHun", "LuHunQu", "LongMenZhen"],
    "Luo": ["LingKou", "LuShi", "ChangShui", "YiYang", "BaiMaSi"],
    "YiLuo": ["HeiShiGuan"],
}
CANALS = {
    "Yi": ["BoYunLingQu", "YueJinQu", "MaoZhuangQu", "LuHunGuangaiQu", "YiDongQu"],
    "Luo": ["LuShiLuoBeiQu", "LuoNingLuoBeiQu", "YiLuoQu", "ZhongZhouQu"],
    "YiLuo": [],
}

SEGMENTS = [
    ("Yi S1: Luanchuan-Tantou",     "Yi",  "LuanChuan",  "TanTou",      ["BoYunLingQu", "YueJinQu"]),
    ("Yi S2: Tantou-Dongwan",       "Yi",  "TanTou",     "DongWan",     []),
    ("Yi S3: Dongwan-Luhun",        "Yi",  "DongWan",    "LuHun",       []),
    ("Yi S4: LuHun-LongMenZhen",    "Yi",  "LuHun",      "LongMenZhen", ["MaoZhuangQu", "LuHunGuangaiQu", "YiDongQu"]),
    ("Luo S1: LingKou-LuShi",       "Luo", "LingKou",    "LuShi",       ["LuShiLuoBeiQu"]),
    ("Luo S2: LuShi-ChangShui",     "Luo", "LuShi",      "ChangShui",   ["LuoNingLuoBeiQu"]),
    ("Luo S3: ChangShui-YiYang",    "Luo", "ChangShui",  "YiYang",      ["YiLuoQu"]),
    ("Luo S4: YiYang-BaiMaSi",      "Luo", "YiYang",     "BaiMaSi",     ["ZhongZhouQu"]),
    ("YiLuo: Confluence-HSG",        "YiLuo", ["LongMenZhen", "BaiMaSi"], "HeiShiGuan", []),
]
# earlier segment definitions, for comparison
LEGACY = {
    "Yi S4: LuHun-LongMenZhen":  ("LuHunQu",   ["MaoZhuangQu", "LuHunGuangaiQu"]),
    "Luo S2: LuShi-ChangShui":   ("LuShi",     ["LuShiLuoBeiQu", "LuoNingLuoBeiQu"]),
    "Luo S3: ChangShui-YiYang":  ("ChangShui", ["LuoNingLuoBeiQu", "YiLuoQu"]),
    "Luo S4: YiYang-BaiMaSi":    ("YiYang",    ["YiLuoQu", "ZhongZhouQu"]),
    "YiLuo: Confluence-HSG":     (["LongMenZhen", "BaiMaSi"], ["YiDongQu"]),
}
FLOOD_PCT = 0.98


def get(rn, name):
    return pd.to_numeric(rn[name], errors="coerce") if name in rn.columns else None


def detect_derived(rn, river):
    chans = [c for c in CHANNEL[river] if c in rn.columns]
    canals = [c for c in CANALS[river] if c in rn.columns]
    rows = []
    for c in chans:
        C = get(rn, c)
        best = None
        for u in chans:
            if u == c:
                continue
            U = get(rn, u)
            for k in range(0, min(3, len(canals)) + 1):
                for S in itertools.combinations(canals, k):
                    pred = U - sum(get(rn, s).fillna(0) for s in S) if S else U.copy()
                    diff = (C - pred).dropna()
                    if len(diff) < 100:
                        continue
                    share = float((diff.abs() <= TOL).mean())
                    cand = dict(station=c, formula=f"{u}" + "".join(f" - {s}" for s in S),
                                n=len(diff), share_exact=100 * share,
                                max_abs=float(diff.abs().max()), rmse=float(np.sqrt((diff ** 2).mean())))
                    if best is None or (cand["share_exact"], -cand["rmse"]) > (best["share_exact"], -best["rmse"]):
                        best = cand
        if best:
            best["verdict"] = ("DERIVED (computed by subtraction)" if best["share_exact"] >= 95
                               else "probably derived" if best["share_exact"] >= 80 else "independent gauge")
            rows.append(best)
    return pd.DataFrame(rows)


def audit(rn, flood_mask=None):
    rows = []
    for label, river, up, dn, canals in SEGMENTS:
        ups = up if isinstance(up, list) else [up]
        U = None
        for u in ups:
            g = get(rn, u)
            if g is None:
                U = None; break
            U = g if U is None else U + g
        D = get(rn, dn)
        if U is None or D is None:
            print(f"  skip {label}: station missing"); continue
        Qdiv = sum(get(rn, c).fillna(0) for c in canals if get(rn, c) is not None) if canals else 0.0
        lup, lcan = LEGACY.get(label, (up, canals))
        LU = U if lup == up else get(rn, lup)
        LQ = sum(get(rn, c).fillna(0) for c in lcan if get(rn, c) is not None) if lcan else 0.0
        keep = D <= D.quantile(FLOOD_PCT)
        cor_all, leg_all = (U - D - Qdiv).dropna(), (LU - D - LQ).dropna()
        cor_nf, leg_nf = (U - D - Qdiv)[keep].dropna(), (LU - D - LQ)[keep].dropna()
        rows.append(dict(segment=label, n_all=len(cor_all), n_noflood=len(cor_nf),
                         flood_threshold=round(float(D.quantile(FLOOD_PCT)), 1),
                         earlier_def_mean_all=leg_all.mean(), mean_all=cor_all.mean(),
                         earlier_def_mean_noflood=leg_nf.mean(), mean_noflood=cor_nf.mean(),
                         median_noflood=cor_nf.median(),
                         pct_neg_noflood=100 * (cor_nf < 0).mean(),
                         note="earlier definition: LuHunQu inflow, offtakes subtracted again" if label in LEGACY and lup != up
                              else ("earlier definition: canal also subtracted in the adjacent segment"
                                    if label in LEGACY and set(lcan) != set(canals) and not label.startswith("YiLuo") else "")))
    return pd.DataFrame(rows)


_NET_CACHE = {}


def _is_net(rn, station, canals):
    key = (station, tuple(sorted(canals)))
    if key in _NET_CACHE:
        return _NET_CACHE[key]
    ok = False
    if canals:
        S = get(rn, station)
        for river, chans in CHANNEL.items():
            for u in chans:
                if u == station or u not in rn.columns:
                    continue
                pred = get(rn, u) - sum(get(rn, c).fillna(0) for c in canals if get(rn, c) is not None)
                diff = (S - pred).dropna()
                if len(diff) >= 100 and (diff.abs() <= TOL).mean() >= 0.95:
                    ok = True
    _NET_CACHE[key] = ok
    return ok


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    wq_daily, rn_wide, precip, ps = mr.load_all()
    rn = rn_wide.copy(); rn.index = pd.to_datetime(rn.index)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

    print("\nderived-series check")
    out = []
    for river in ("Yi", "Luo", "YiLuo"):
        if not [c for c in CHANNEL[river] if c in rn.columns]:
            continue
        t = detect_derived(rn, river)
        if len(t):
            t.insert(0, "river", river)
            print(f"\n{river}")
            print(t.round(3).to_string(index=False))
            out.append(t)
    if out:
        pd.concat(out).to_csv(f"{OUTDIR}/derived_series_check.csv", index=False, encoding="utf-8-sig")

    print("\nsegment flow balance (noflood = downstream gauge <= P98)")
    a = audit(rn)
    print(a.round(2).to_string(index=False))
    a.to_csv(f"{OUTDIR}/flow_audit_segments.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved {OUTDIR}/derived_series_check.csv, {OUTDIR}/flow_audit_segments.csv")


if __name__ == "__main__":
    main()
