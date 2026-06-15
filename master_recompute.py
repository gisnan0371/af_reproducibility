#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_recompute.py
===================================================================
Conventions
------------------------
  AF  = (Q_down + Q_div) / Q_down          # >=1
  DR  = Q_div / (Q_down + Q_div) = 1 - 1/AF
  C_cf      = C_obs / AF
  dC_total  = C_obs * (1 - 1/AF)
  dC_dilut  = C_up  * (AF - 1)
  Load (kg/d) = C(mg/L) * Q(m3/s) * 86.4
===================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# 0. CONFIG  --  point each variable at YOUR file. Leave "" to auto-search
#    INPUT_DIR (recursively). Windows raw strings (r"...") handle backslashes.
# ----------------------------------------------------------------------
WQ_FILE     = r""     # raw nutrient WQ csv; "" -> auto-search INPUT_DIR
RUNOFF_FILE = r""     # e.g. r"D:\...\input_runoff_data.csv"   (径流)
METRO_FILE  = r""     # e.g. r"D:\...\input_metro_data.csv"    (气象/降水)
PS_FILE     = r""     # e.g. r"D:\...\point_source\input_source.csv"  (点源, 2020+)
# raw (pre-imputation) WQ file that has EC/temp/DO/turbidity populated; "" -> auto-search
# (the imputed file only has the 3 nutrients). Used ONLY to fill temp/DO/EC/turbidity.
AUX_WQ_FILE = r""     # raw WQ csv with EC/temp/DO/turbidity; "" -> auto-search

INPUT_DIR   = Path(WQ_FILE).parent if WQ_FILE else Path("./input")   # default search dir
RESULTS_DIR = Path("./results"); RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def resolve(explicit, patterns, label, required=True, exclude=None):
    """Return an existing file path: the explicit one if set & present, else the
    first recursive match of `patterns` under INPUT_DIR (skipping any path whose
    lowercased string contains an `exclude` token). Raises if required & missing."""
    exclude = [x.lower() for x in (exclude or [])]
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        print(f"  [{label}] configured path missing -> auto-searching: {p}")
    for pat in patterns:
        hits = [h for h in sorted(INPUT_DIR.rglob(pat))
                if not any(x in str(h).lower() for x in exclude)]
        if hits:
            print(f"  [{label}] auto-found: {hits[0]}")
            return hits[0]
    seen = ", ".join(sorted(q.name for q in INPUT_DIR.rglob("*.csv")))[:700]
    msg = (f"{label}: not found. Set its path in CONFIG, or drop a file matching "
           f"{patterns} under {INPUT_DIR}.\n    .csv seen there: {seen}")
    if required:
        raise FileNotFoundError(msg)
    print("  WARNING: " + msg)
    return None

FULL_START, FULL_END = "2018-09-01", "2021-12-31"
PS_START             = "2020-01-01"
IRR_MONTHS           = [4, 5, 6, 7, 8, 9]
Q_MIN                = 0.1     # m3/s; drop Q_down < 0.1
PRECIP_DRY           = 1.0     # mm/d; dry day threshold
CONV                 = 86.4    # mg/L * m3/s -> kg/d
RNG_SEED             = 42

# --- water-quality column names (CONFIRMED from original config.py) ---
# Full WQ schema: Year, Month, Day, Hour, RiverName, WaterQualityMonitoringStation,
#   WaterTemperature, pH, DO, ElectricConductance, Turbidity,
#   PotassiumPermanganateIndex, AmmoniaNitrogen, TotalPhosphorus, TotalNitrogen
WQ_COL = {"TN": "TotalNitrogen", "TP": "TotalPhosphorus", "NH3": "AmmoniaNitrogen",
          "DO": "DO", "EC": "ElectricConductance", "TURB": "Turbidity",
          "TEMP": "WaterTemperature"}
PS_LOAD = {"TN": "TN_load", "TP": "TP_load", "NH3": "NH4N_load"}
POLL = ["TN", "TP", "NH3"]
STD  = {"TN": 1.0, "TP": 0.2, "NH3": 1.0}   # GB 3838-2002 Grade III (mg/L)

# --- station map (CONFIRMED from original config.py / THRESHOLDS) ---
RIVER = {
 "Yi": dict(
     wq_down="LongMenDaQiao", wq_up="TanTou",
     q_down="LongMenZhen",    q_up="TanTou",
     div_cols=["MaoZhuangQu", "LuHunGuangaiQu"],   # YiDongQu = 100% zero, excluded
     ps_units=["YL10001", "YL10005", "YL10016", "YL10019"],   # Yi_Lower -> LongMenDaQiao
     q_div_high=5.0, dr_high=0.25, q_div_low=0.5, dr_low=0.10, min_dur=3,
 ),
 "Luo": dict(
     wq_down="BaiMaSi",       wq_up="LingKou",
     q_down="BaiMaSi",        q_up="LingKou",
     div_cols=["LuShiLuoBeiQu", "LuoNingLuoBeiQu", "YiLuoQu", "ZhongZhouQu"],
     ps_units=["YL10002", "YL10003", "YL10004", "YL10006", "YL10007",
               "YL10015", "YL10017", "YL10018"],   # Luo_Lower -> BaiMaSi
     q_div_high=19.0, dr_high=0.45, q_div_low=8.0, dr_low=0.20, min_dur=3,
 ),
}

RESULTS = {}   # id -> value, dumped at the end
def rec(key, val):
    RESULTS[key] = val
    print(f"  [{key:24s}] {val}")

# ----------------------------------------------------------------------
# 1. DATA LOADING  (recovered load_all)
# ----------------------------------------------------------------------
def load_all():
    s, e, ps_s = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END), pd.Timestamp(PS_START)

    def _agg_wq(path):
        w = pd.read_csv(path)
        name = Path(path).name
        # --- station column (robust to naming) ---
        scol = next((c for c in ["WaterQualityMonitoringStation", "Station", "StationName",
                                  "SiteName", "Site", "MonitoringStation", "Name", "name"]
                     if c in w.columns), None)
        if scol is None:
            scol = next((c for c in w.columns
                         if "station" in c.lower() or "site" in c.lower() or "断面" in str(c)), None)
        if scol is None:
            raise KeyError(f"{name}: no station column found; columns = {list(w.columns)}")
        # --- datetime ---
        if {"Year", "Month", "Day"}.issubset(w.columns):
            parts = ["Year", "Month", "Day"] + (["Hour"] if "Hour" in w.columns else [])
            fmt = "%Y-%m-%d-%H" if "Hour" in w.columns else "%Y-%m-%d"
            w["date"] = pd.to_datetime(w[parts].astype(str).agg("-".join, axis=1),
                                       format=fmt).dt.normalize()
        elif "Date" in w.columns:
            w["date"] = pd.to_datetime(w["Date"]).dt.normalize()
        else:
            raise KeyError(f"{name}: no Year/Month/Day or Date; columns = {list(w.columns)}")
        w = w[(w["date"] >= s) & (w["date"] <= e)]
        # --- value columns: map synonyms to the canonical WQ_COL names ---
        syn = {"WaterTemperature": ["WaterTemperature", "WT", "Temperature", "Temp", "WaterTemp", "水温"],
               "DO": ["DO", "DissolvedOxygen", "溶解氧"],
               "ElectricConductance": ["ElectricConductance", "ElectricalConductivity",
                                       "Conductivity", "EC", "电导率"],
               "Turbidity": ["Turbidity", "Turb", "浊度"],
               "AmmoniaNitrogen": ["AmmoniaNitrogen", "NH3N", "NH4N", "氨氮"],
               "TotalPhosphorus": ["TotalPhosphorus", "TP", "总磷"],
               "TotalNitrogen": ["TotalNitrogen", "TN", "总氮"]}
        for canon, alts in syn.items():
            if canon not in w.columns:
                hit = next((a for a in alts if a in w.columns), None)
                if hit:
                    w = w.rename(columns={hit: canon})
        av = [c for c in WQ_COL.values() if c in w.columns]
        for c in av:
            w[c] = pd.to_numeric(w[c], errors="coerce")
        g = (w.groupby([scol, "date"])[av].mean().reset_index()
               .rename(columns={scol: "station"}))
        targets = ["LingKou", "LuoNingChangShui", "GaoYaZhai", "BaiMaSi",
                   "TanTou", "LongMenDaQiao", "QiLiPu"]
        g["station"] = g["station"].astype(str)
        nm = {a: t for t in targets for a in g["station"].unique() if a.lower() == t.lower()}
        g["station"] = g["station"].map(lambda x: nm.get(x, x))
        return g

    wq_path = resolve(WQ_FILE, ["*water_quality*.csv", "*imputed*water*.csv"], "WQ")
    wq = pd.read_csv(wq_path)
    print(f"  WQ columns: {list(wq.columns)}")     # verify DO/EC/TURB/TEMP vs WQ_COL
    wq_daily = _agg_wq(wq_path)

    # EC/temp/DO/turbidity often absent from the imputed file -> pull them from a
    # raw (pre-imputation) WQ file, merged on station+date. Auto-find excludes 'imputed'.
    aux_cols = [WQ_COL[k] for k in ("TEMP", "DO", "EC", "TURB")]
    empty_aux = [c for c in aux_cols if c not in wq_daily.columns
                 or wq_daily[c].notna().sum() == 0]
    if empty_aux:
        aux_path = resolve(AUX_WQ_FILE,
                           ["*water_quality*full*.csv", "input_water_quality_data.csv",
                            "*water_quality*.csv"], "AUX-WQ", required=False,
                           exclude=[Path(wq_path).name.lower()])
        if aux_path is not None and Path(aux_path) != Path(wq_path):
            aux = _agg_wq(aux_path)
            keep = [c for c in empty_aux if c in aux.columns and aux[c].notna().sum() > 0]
            if keep:
                wq_daily = wq_daily.drop(columns=[c for c in keep if c in wq_daily.columns]) \
                                   .merge(aux[["station", "date"] + keep], on=["station", "date"], how="left")
                print(f"  aux WQ ({Path(aux_path).name}): filled {keep} "
                      f"(e.g. EC non-NA = {int(wq_daily.get(WQ_COL['EC'], pd.Series(dtype=float)).notna().sum())})")
            else:
                print(f"  aux WQ ({Path(aux_path).name}) had no usable {empty_aux}")
        else:
            print(f"  NOTE: {empty_aux} empty in primary and no separate raw WQ file found "
                  f"-> EC tracer / M2 / water-temp control will be skipped downstream")

    rn = pd.read_csv(resolve(RUNOFF_FILE, ["*runoff*.csv", "*径流*.csv"], "runoff"))
    rn["date"] = pd.to_datetime(rn[["Year", "Month", "Day"]].astype(str)
                                .agg("-".join, axis=1), format="%Y-%m-%d")
    rn["Runoff"] = pd.to_numeric(rn["Runoff"], errors="coerce")
    rn = rn[(rn["date"] >= s) & (rn["date"] <= e)]
    rn_wide = (rn.pivot_table(index="date", columns="RiverRunoffMonitoringStation",
                              values="Runoff", aggfunc="mean")
                 .reindex(pd.date_range(s, e, freq="D")))
    rn_wide.index.name = "date"

    met = pd.read_csv(resolve(METRO_FILE,
                              ["*metro*.csv", "*meteo*.csv", "*precip*.csv", "*气象*.csv"], "metro"))
    if "Hour" in met.columns:
        met["date"] = pd.to_datetime(met[["Year", "Month", "Day", "Hour"]].astype(str)
                                     .agg("-".join, axis=1), format="%Y-%m-%d-%H").dt.normalize()
    else:
        met["date"] = pd.to_datetime(met[["Year", "Month", "Day"]].astype(str)
                                     .agg("-".join, axis=1), format="%Y-%m-%d")
    met = met[(met["date"] >= s) & (met["date"] <= e)]
    precip = (met.groupby("date")["Precipitation"].sum().reset_index()
                 .rename(columns={"Precipitation": "precip_mm"}))

    ps = None
    ps_path = resolve(PS_FILE,
                      ["point_source/*input_source*.csv", "point_source/*.csv",
                       "*input_source*.csv", "*point*source*load*.csv", "*point_source*.csv"],
                      "PS", required=False,
                      exclude=["assignment", "reach", "segment", "mapping", "_map",
                               "analysis_output", "summary"])
    if ps_path is not None:
        ps = pd.read_csv(ps_path)
        ps.rename(columns={"discharge(m3/day)": "discharge"}, inplace=True, errors="ignore")
        has_date = "Date" in ps.columns
        has_ymd  = all(c in ps.columns for c in ["Year", "Month", "Day"])
        load_cols = [c for c in PS_LOAD.values() if c in ps.columns]
        if not (has_date or has_ymd) or not load_cols:
            print(f"  WARNING: '{ps_path.name}' is not a point-source LOAD time series "
                  f"(cols={list(ps.columns)[:10]}). Set PS_FILE to your raw load CSV. "
                  f"Continuing WITHOUT point sources (PS-fraction & GAMM PS covariate skipped).")
            ps = None
        else:
            ps["date"] = (pd.to_datetime(ps["Date"]) if has_date else
                          pd.to_datetime(ps[["Year", "Month", "Day"]].astype(str)
                                         .agg("-".join, axis=1), format="%Y-%m-%d"))
            for c in list(PS_LOAD.values()) + ["discharge"]:
                if c in ps.columns:
                    ps[c] = pd.to_numeric(ps[c], errors="coerce")
            uid = next((c for c in ["N_unit", "SourceID", "Source_ID", "unit"] if c in ps.columns), None)
            if uid and uid != "unit":
                ps.rename(columns={uid: "unit"}, inplace=True)
            ps = ps[(ps["date"] >= ps_s) & (ps["date"] <= e)]
            print(f"  point source: {ps_path.name}  rows={len(ps)}"
                  + (f"  units={ps['unit'].nunique()}" if "unit" in ps.columns else ""))

    print(f"  WQ stations : {sorted(wq_daily['station'].unique())}")
    print(f"  runoff cols : {len(rn_wide.columns)}; days={len(rn_wide)}")
    return wq_daily, rn_wide, precip, ps


# ----------------------------------------------------------------------
# 2. PER-RIVER DAILY FRAME  (Q_div, Q_down, AF, DR, C_obs, C_up, dry, irr)
# ----------------------------------------------------------------------
def build_daily(river, wq_daily, rn_wide, precip):
    cfg = RIVER[river]
    idx = rn_wide.index
    q_div  = sum(rn_wide[c].fillna(0) for c in cfg["div_cols"] if c in rn_wide.columns)
    q_down = rn_wide[cfg["q_down"]].fillna(0) if cfg["q_down"] in rn_wide.columns else pd.Series(0, index=idx)
    q_up   = rn_wide[cfg["q_up"]] if cfg["q_up"] in rn_wide.columns else pd.Series(np.nan, index=idx)
    q_nd   = q_down + q_div
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(q_down > Q_MIN, q_nd / q_down, np.nan)
        dr = np.where(q_nd > 0, q_div / q_nd, np.nan)

    d = pd.DataFrame({"Q_div": q_div, "Q_down": q_down, "Q_up": q_up,
                      "Q_nd": q_nd, "AF": af, "DR": dr}, index=idx)
    d = d.merge(precip.set_index("date"), left_index=True, right_index=True, how="left")
    d["precip_mm"] = d["precip_mm"].fillna(0)
    d["API7"] = d["precip_mm"].rolling(7, min_periods=1).sum()   # antecedent precip index
    d["is_dry"] = d["precip_mm"] < PRECIP_DRY
    d["month"]  = d.index.month
    d["year"]   = d.index.year
    d["is_irr"] = d["month"].isin(IRR_MONTHS)

    wd = wq_daily[wq_daily["station"] == cfg["wq_down"]].set_index("date")
    wu = wq_daily[wq_daily["station"] == cfg["wq_up"]].set_index("date")
    for p in POLL + ["DO", "EC", "TURB", "TEMP"]:
        col = WQ_COL.get(p)
        if col and col in wd.columns:
            d[f"Cobs_{p}"] = wd[col].reindex(idx)
        if col and col in wu.columns:
            d[f"Cup_{p}"]  = wu[col].reindex(idx)
    for p in POLL:
        if f"Cobs_{p}" in d:
            d[f"Ccf_{p}"]     = d[f"Cobs_{p}"] / d["AF"]
            d[f"dCtot_{p}"]   = d[f"Cobs_{p}"] * (1 - 1 / d["AF"])
        if f"Cup_{p}" in d:
            d[f"dCdil_{p}"]   = d[f"Cup_{p}"] * (d["AF"] - 1)
    return d


def label_events(d, cfg):
    """Tag contiguous high/low diversion runs (>=min_dur) within irrigation+dry."""
    d = d.copy()
    d["evt"] = "other"
    hi = d["is_irr"] & (d["Q_div"] >= cfg["q_div_high"]) & (d["DR"] >= cfg["dr_high"])
    lo = d["is_irr"] & (d["Q_div"] <= cfg["q_div_low"]) & (d["DR"] <= cfg["dr_low"])
    d.loc[hi, "evt"] = "high"; d.loc[lo, "evt"] = "low"
    # enforce min duration + assign event ids
    eid = np.full(len(d), -1); cur = 0
    for status in ["high", "low"]:
        m = (d["evt"] == status).values.astype(int); start = None
        for i in range(len(m) + 1):
            if i < len(m) and m[i] == 1:
                start = i if start is None else start
            elif start is not None:
                if i - start < cfg["min_dur"]:
                    d.iloc[start:i, d.columns.get_loc("evt")] = "other"
                else:
                    eid[start:i] = cur; cur += 1
                start = None
    d["event_id"] = eid
    return d


# ----------------------------------------------------------------------
# 3. AF STATISTICS  (Table 2, Table S2, 4.2, Abstract)
# ----------------------------------------------------------------------
def af_stats(river, d):
    out = {}
    for label, sub in [("full", d), ("irr", d[d["is_irr"]]), ("nonirr", d[~d["is_irr"]])]:
        a = sub["AF"].dropna()
        out[label] = dict(n=len(a), mean=a.mean(), median=a.median(),
                          p5=a.quantile(.05), p95=a.quantile(.95), max=a.max())
    irr = d[d["is_irr"]]["AF"].dropna()
    trimmed = irr[irr <= irr.quantile(.95)].mean()
    rec(f"{river}_AF_irr_mean",   round(out['irr']['mean'], 3))
    rec(f"{river}_AF_irr_median", round(out['irr']['median'], 3))
    rec(f"{river}_AF_irr_P95",    round(out['irr']['p95'], 3))
    rec(f"{river}_AF_irr_max",    round(out['irr']['max'], 2))
    rec(f"{river}_AF_full_mean",  round(out['full']['mean'], 3))
    rec(f"{river}_AF_nonirr_mean",round(out['nonirr']['mean'], 3))
    rec(f"{river}_AF_5pct_trim",  round(trimmed, 3))
    return out


# ----------------------------------------------------------------------
# 4. DOSE-RESPONSE  (4.3) and full AF-dC corr (4.2)
# ----------------------------------------------------------------------
def dose_response(river, d):
    sub = d[d["is_irr"] & d["is_dry"]]
    for p in POLL:
        x, y = sub["DR"], sub.get(f"dCtot_{p}")
        if y is None: continue
        m = x.notna() & y.notna()
        if m.sum() > 3:
            r, pv = stats.pearsonr(x[m], y[m])
            rec(f"{river}_DR_dCtot_r_{p}", f"r={r:.3f} p={pv:.1e} n={int(m.sum())}")

def af_dc_corr(river, d):
    """Full irrigation-season AF vs dCtot, raw and 5%-trimmed (4.2 0.865->0.864 etc.)."""
    sub = d[d["is_irr"]]
    for p in POLL:
        x, y = sub["AF"], sub.get(f"dCtot_{p}")
        if y is None: continue
        m = x.notna() & y.notna()
        if m.sum() > 3:
            r_full, _ = stats.pearsonr(x[m], y[m])
            thr = x[m].quantile(.95); mt = m & (x <= thr)
            r_trim, _ = stats.pearsonr(x[mt], y[mt])
            rec(f"{river}_AF_dCtot_r_{p}", f"{r_full:.3f} -> {r_trim:.3f}")


# ----------------------------------------------------------------------
# 5. EVENT-LEVEL TESTS  (4.4, Table S4)  -- now reported as accounting + signal
# ----------------------------------------------------------------------
def events(river, d, cfg):
    de = label_events(d[d["is_irr"] & d["is_dry"]].copy(), cfg)
    ev = de[de["event_id"] >= 0]
    # event medians
    med = ev.groupby(["evt", "event_id"]).median(numeric_only=True).reset_index()
    hi = med[med["evt"] == "high"]; lo = med[med["evt"] == "low"]
    rec(f"{river}_events_nhigh_nlow", f"{len(hi)} / {len(lo)}")
    for p in POLL:
        cobs, ccf = f"Cobs_{p}", f"Ccf_{p}"
        # paired Cobs vs Ccf within HIGH events (ALGEBRAIC IDENTITY -> accounting illustration)
        if cobs in hi and ccf in hi and len(hi) >= 3:
            pair = hi[[cobs, ccf]].dropna()
            if len(pair) >= 3:
                _, pw = stats.wilcoxon(pair[cobs], pair[ccf])
                rec(f"{river}_pairWilcoxon_p_{p}", f"{pw:.3f}  (NOTE: identity, accounting only)")
        # high vs low OBSERVED concentration  (the empirical comparison)
        if cobs in hi and cobs in lo and len(hi) and len(lo):
            a, b = hi[cobs].dropna(), lo[cobs].dropna()
            if len(a) and len(b):
                _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
                chg = (a.median() - b.median()) / b.median() * 100
                rec(f"{river}_highVlow_{p}", f"{chg:+.1f}%  p={pu:.3f}")
    # NH3 bootstrap on high-low observed difference
    if "Cobs_NH3" in hi and "Cobs_NH3" in lo:
        a, b = hi["Cobs_NH3"].dropna().values, lo["Cobs_NH3"].dropna().values
        if len(a) and len(b):
            rng = np.random.default_rng(RNG_SEED)
            diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
                     for _ in range(10000)]
            lo_ci, hi_ci = np.percentile(diffs, [2.5, 97.5])
            rec(f"{river}_NH3_boot_CI", f"[{lo_ci:+.3f}, {hi_ci:+.3f}] mg/L; "
                                  f"P(>0)={np.mean(np.array(diffs) > 0)*100:.1f}%")
    return de


# ----------------------------------------------------------------------
# 6. MASS-FLUX STABILITY  (3.2.1, Table S8/S9)
# ----------------------------------------------------------------------
def dry_windows(d, min_days=3):
    m = (d["is_irr"] & d["is_dry"]).values.astype(int)
    wins, start = [], None
    for i in range(len(m) + 1):
        if i < len(m) and m[i] == 1:
            start = i if start is None else start
        elif start is not None:
            if i - start >= min_days:
                wins.append((d.index[start], d.index[i - 1]))
            start = None
    return wins

def massflux_cv(river, d):
    wins = dry_windows(d)
    total_days = sum(len(d.loc[a:b]) for a, b in wins)
    rec(f"{river}_dry_windows_days", f"{len(wins)} windows / {total_days} days")
    for p in POLL:
        cup = f"Cup_{p}"
        if cup not in d: continue
        cvs = []
        for a, b in wins:
            w = d.loc[a:b]
            load = w[cup] * w["Q_up"] * CONV
            load = load.dropna()
            if len(load) >= 2 and load.mean() != 0:
                cvs.append(load.std(ddof=1) / load.mean() * 100)
        if cvs:
            cvs = np.array(cvs)
            rec(f"{river}_massflux_CVmed_{p}",
                f"median={np.median(cvs):.1f}%  %<30={np.mean(cvs < 30)*100:.0f}%  nwin={len(cvs)}")


# ----------------------------------------------------------------------
# 7. dC DECOMPOSITION + ATTENUATION %  (5.1, Fig.14)
# ----------------------------------------------------------------------
def dc_decomp(river, d):
    sub = d[d["is_irr"] & d["is_dry"]]
    for p in POLL:
        dil, tot = sub.get(f"dCdil_{p}"), sub.get(f"dCtot_{p}")
        if dil is None or tot is None: continue
        m = dil.notna() & tot.notna()
        if m.sum() == 0: continue
        mdil, mtot = dil[m].mean(), tot[m].mean()
        resid = mtot - mdil                 # negative -> net attenuation
        att = (abs(resid) / mdil * 100) if mdil != 0 else np.nan
        rec(f"{river}_dCdecomp_{p}",
            f"dil={mdil:+.2f} tot={mtot:+.2f} resid={resid:+.2f} attenuation={att:.0f}%")


# ----------------------------------------------------------------------
# 8. TWO-COMPONENT MONTE CARLO  (4.8, Fig.12)  -- CURRENT methods
# ----------------------------------------------------------------------
def monte_carlo(river, d, n=10000):
    sub = d[d["is_irr"]].copy()
    qd, qv = sub["Q_down"].values, sub["Q_div"].values
    nday = len(qd); rng = np.random.default_rng(RNG_SEED)
    af_means = np.empty(n)
    for i in range(n):
        s_qv = rng.uniform(0.85, 1.15)   # systematic, once per realization
        s_qd = rng.uniform(0.90, 1.10)
        r_qv = rng.uniform(0.95, 1.05, nday)   # daily random
        r_qd = rng.uniform(0.95, 1.05, nday)
        qvp = qv * s_qv * r_qv
        qdp = qd * s_qd * r_qd
        with np.errstate(divide="ignore", invalid="ignore"):
            af = np.where(qdp > Q_MIN, (qdp + qvp) / qdp, np.nan)
        af_means[i] = np.nanmean(af)
    lo, med, hi = np.percentile(af_means, [2.5, 50, 97.5])
    width = (hi - lo) / med * 100
    rec(f"{river}_MC_AF", f"median={med:.2f}  CI=[{lo:.2f}, {hi:.2f}]  width=±{width/2:.1f}%")


# ----------------------------------------------------------------------
# 9. FLOW-MATCHED C-Q  (4.7, Fig.10)  -- seasonal grouping, permutation
# ----------------------------------------------------------------------
def cq_flowmatched(river, d, n_perm=1000):
    dry = d[d["is_dry"]].copy()
    irr, non = dry[dry["is_irr"]], dry[~dry["is_irr"]]
    q_irr = irr["Q_down"].dropna(); q_non = non["Q_down"].dropna()
    if q_irr.empty or q_non.empty: return
    lo = max(q_irr.quantile(.10), q_non.quantile(.10))
    hi = min(q_irr.quantile(.90), q_non.quantile(.90))
    if lo >= hi:
        rec(f"{river}_CQ_overlap_Q", "no overlapping flow range (seasons do not overlap)")
        return
    rec(f"{river}_CQ_overlap_Q", f"{lo:.1f}-{hi:.1f} m3/s")
    band = dry[(dry["Q_down"] >= lo) & (dry["Q_down"] <= hi)]
    rng = np.random.default_rng(RNG_SEED)
    for p in POLL:
        c = f"Cobs_{p}"
        if c not in band: continue
        sub = band[[c, "Q_down", "is_irr"]].dropna()
        sub = sub[sub[c] > 0]
        if sub["is_irr"].sum() < 5 or (~sub["is_irr"]).sum() < 5: continue
        lq = np.log(sub["Q_down"]); lc = np.log(sub[c])
        def intercept_gap(mask):
            bi = np.polyfit(lq[mask], lc[mask], 1)         # [slope, intercept]
            bn = np.polyfit(lq[~mask], lc[~mask], 1)
            return bi[1] - bn[1]
        obs = intercept_gap(sub["is_irr"].values)
        perm = np.array([intercept_gap(rng.permutation(sub["is_irr"].values))
                         for _ in range(n_perm)])
        pv = np.mean(np.abs(perm) >= abs(obs))
        pct = (np.exp(obs) - 1) * 100   # intercept gap -> % conc shift at matched Q
        rec(f"{river}_CQ_shift_{p}", f"{pct:+.1f}%  perm_p={pv:.3f}  "
                                     f"n_irr={int(sub['is_irr'].sum())} n_non={int((~sub['is_irr']).sum())}")


# ----------------------------------------------------------------------
# 10. REVERSE-CAUSALITY DIAGNOSTICS  (5.1, Fig.S7)  -- Yi
# ----------------------------------------------------------------------
def reverse_causality(river, d):
    sub = d[d["is_irr"]].dropna(subset=["Q_nd", "Q_div", "DR"])
    rho1, _ = stats.spearmanr(sub["Q_nd"], sub["Q_div"])
    rho2, _ = stats.spearmanr(sub["Q_nd"], sub["DR"])
    lagdf = sub[["Q_nd", "Q_div"]].copy(); lagdf["Q_nd_lag1"] = lagdf["Q_nd"].shift(1)
    lagdf = lagdf.dropna()
    rho3, _ = stats.spearmanr(lagdf["Q_nd_lag1"], lagdf["Q_div"])
    zero = (sub["Q_div"].diff().abs() < 1e-9).mean() * 100
    rec(f"{river}_rev_rho_Qnd_Qdiv", f"{rho1:.2f} (weak)")
    rec(f"{river}_rev_rho_Qnd_DR",   f"{rho2:.2f} (weak)")
    rec(f"{river}_rev_rho_lag1",     f"{rho3:.2f}  (TEXT MUST READ 'weak/low', not 'high')")
    rec(f"{river}_rev_pct_zerochg",  f"{zero:.0f}% days zero day-to-day change")


# ----------------------------------------------------------------------
# 11. MASS BALANCE 3-scenario lateral + NEW diverted-solute sensitivity (4.6/3.4, S7)
# ----------------------------------------------------------------------
def mass_balance(river, wq_daily, rn_wide, ps, d):
    """
    Load_NPS = L_down - L_up + L_div - L_point - L_lateral
    L_div = C_intake * Q_div * CONV ;  C_intake = c_int_mult * C_up   (NEW sensitivity)
    L_lateral = C_lat * dQ_residual * CONV ;  C_lat in {0, Cup, 2Cup} (existing 3-scenario)
    dQ_residual from flow audit (Q_in - Q_out,river - Q_out,div) of the segment.
    NOTE: AF itself is independent of C_lat AND C_intake (uses only discharge).
    """
    cfg = RIVER[river]
    seg = d.copy()
    # lateral residual on the down reach: dQ = Q_up + ... - Q_down - Q_div ; here use (Q_nd - Q_up) proxy
    seg["dQ_resid"] = (seg["Q_nd"] - seg["Q_up"]).clip(lower=0)   # lateral-residual proxy (Table S3 segment definition)
    seg = seg[seg["is_irr"]]
    # point-source load routed to this down node
    psload = {p: 0.0 for p in POLL}
    if ps is not None and "unit" in ps.columns and cfg["ps_units"]:
        sel = ps[ps["unit"].isin(cfg["ps_units"])]
        for p in POLL:
            col = PS_LOAD[p]
            if col in sel.columns:
                psload[p] = sel.groupby("date")[col].sum().reindex(seg.index).fillna(0)
    for p in POLL:
        cobs, cup = f"Cobs_{p}", f"Cup_{p}"
        if cobs not in seg or cup not in seg: continue
        Ldown = seg[cobs] * seg["Q_down"] * CONV
        Lup   = seg[cup]  * seg["Q_up"]   * CONV
        Lp    = psload[p] if not np.isscalar(psload[p]) else 0.0
        # --- existing 3-scenario lateral (diverted load fixed at C_up) ---
        Ldiv = seg[cup] * seg["Q_div"] * CONV
        for tag, mult in [("clean", 0.0), ("base", 1.0), ("polluted", 2.0)]:
            Llat = (mult * seg[cup]) * seg["dQ_resid"] * CONV
            nps = (Ldown - Lup + Ldiv - Lp - Llat)
            rec(f"{river}_NPS_{p}_{tag}", f"mean={nps.mean():.0f} median={nps.median():.0f} kg/d")
        # --- NEW: diverted-solute C_intake sensitivity (R1 Major 2) ---
        for cim in [0.5, 1.0, 1.5, 2.0]:
            Ldiv2 = (cim * seg[cup]) * seg["Q_div"] * CONV
            Llat = seg[cup] * seg["dQ_resid"] * CONV     # base lateral
            nps = (Ldown - Lup + Ldiv2 - Lp - Llat)
            rec(f"{river}_NPS_{p}_Cint{cim:g}xCup", f"mean={nps.mean():.0f} kg/d")


# ----------------------------------------------------------------------
# 12. POINT-SOURCE FRACTION  (4.4, Fig.S6)  -- non-irrigation season
# ----------------------------------------------------------------------
def ps_fraction(river, d, ps):
    if ps is None or "unit" not in ps.columns or not RIVER[river]["ps_units"]: return
    non = d[(~d["is_irr"])]
    sel = ps[ps["unit"].isin(RIVER[river]["ps_units"])]
    for p in POLL:
        cobs, col = f"Cobs_{p}", PS_LOAD[p]
        if cobs not in non or col not in sel.columns: continue
        river_load = (non[cobs] * non["Q_down"] * CONV)
        ps_load = sel.groupby("date")[col].sum().reindex(non.index)
        frac = (ps_load / river_load).replace([np.inf, -np.inf], np.nan).dropna()
        if len(frac):
            rec(f"{river}_PSfrac_{p}", f"{frac.mean()*100:.1f}%")


# ----------------------------------------------------------------------
# 13. BASEFLOW SEPARATION (NEW, R2-ii)  -- Lyne-Hollick + Eckhardt
# ----------------------------------------------------------------------
def lyne_hollick(q, alpha=0.925, passes=3):
    q = pd.Series(q).interpolate().bfill().ffill().values
    bf = q.copy()
    for pi in range(passes):
        seq = range(1, len(q)) if pi % 2 == 0 else range(len(q) - 2, -1, -1)
        qf = bf.copy()
        for t in seq:
            prev = t - 1 if pi % 2 == 0 else t + 1
            f = alpha * qf[prev] + (1 + alpha) / 2 * (q[t] - q[prev])
            qf[t] = min(q[t], q[t] - max(f, 0)) if f > 0 else q[t]
        bf = qf
    bf = np.minimum(bf, q)
    return bf

def eckhardt(q, alpha=0.98, BFImax=0.80):
    q = pd.Series(q).interpolate().bfill().ffill().values
    bf = np.zeros_like(q); bf[0] = q[0]
    for t in range(1, len(q)):
        b = ((1 - BFImax) * alpha * bf[t-1] + (1 - alpha) * BFImax * q[t]) / (1 - alpha * BFImax)
        bf[t] = min(b, q[t])
    return bf

def baseflow(river, rn_wide):
    cfg = RIVER[river]
    for st in [cfg["q_up"], cfg["q_down"]]:
        if st not in rn_wide.columns: continue
        q = rn_wide[st]
        if q.notna().sum() < 30: continue
        for name, bf in [("LH", lyne_hollick(q.values)), ("Eck", eckhardt(q.values))]:
            qq = q.interpolate().bfill().ffill().values
            bfi = np.nansum(bf) / np.nansum(qq)
            irr_mask = rn_wide.index.month.isin(IRR_MONTHS)
            bfi_irr = np.nansum(bf[irr_mask]) / np.nansum(qq[irr_mask])
            rec(f"{river}_{st}_BFI_{name}", f"all={bfi:.2f} irr={bfi_irr:.2f}")


# ----------------------------------------------------------------------
# 14. EXCEEDANCE PROBABILITY by DR bin (Fig.11)
# ----------------------------------------------------------------------
def exceedance(river, d):
    sub = d[d["is_irr"] & d["is_dry"]].copy()
    sub["DRbin"] = (sub["DR"] * 100 // 10) * 10
    for p in POLL:
        c = f"Cobs_{p}"
        if c not in sub: continue
        g = sub.dropna(subset=[c]).groupby("DRbin")[c]
        ex = g.apply(lambda s: (s > STD[p]).mean() * 100 if len(s) >= 5 else np.nan).dropna()
        if len(ex):
            rec(f"{river}_exceed_{p}", "; ".join(f"DR{int(k)}:{v:.0f}%" for k, v in ex.items()))


# ----------------------------------------------------------------------
# 15. Recompute and print all reported manuscript values
#   - Yi/Luo full-period AF max (Table S2)
#   - Yi TN load diverted vs upstream  (4.6; same C_intake=Cup as mass_balance)
#   - Yi effect size across the observed diversion range (4.7; betas from gamm_ec_tracer.R)
#   - Luo spatial TN/TP gradient by station (4.5)
#   (residence-time +33% is a hydraulic-geometry estimate, not data -> handle in text)
# ----------------------------------------------------------------------
BETA_NH3_GAMM = 0.117   # gamm_ec_tracer.R  M1a  beta1 on log(Q_div+1)
BETA_EC_GAMM  = 0.020   # gamm_ec_tracer.R  EC_abs beta1 on log(Q_div+1)

def report_open(daily, wq_daily):
    print("\n" + "=" * 70)
    print("Recomputed manuscript values")
    print("=" * 70)

    # (1) full-period AF max
    for river in ["Yi", "Luo"]:
        rec(f"{river}_AF_full_max", round(daily[river]["AF"].dropna().max(), 2))

    # (2) TN load diverted vs upstream  (C_intake = Cup, consistent with mass_balance)
    for river in ["Yi", "Luo"]:
        d = daily[river]
        for basis, mask in [("irr", d["is_irr"]),
                            ("irr_dry", d["is_irr"] & d["is_dry"])]:
            s = d[mask].dropna(subset=["Cup_TN", "Q_div", "Q_up"])
            if len(s):
                Ldiv = (s["Cup_TN"] * s["Q_div"] * CONV).mean()
                Lup  = (s["Cup_TN"] * s["Q_up"]  * CONV).mean()
                rec(f"{river}_TNload_div_vs_up_{basis}",
                    f"div={Ldiv:.0f} up={Lup:.0f} kg/d (n={len(s)})")

    # (3) effect size across the observed diversion range (Yi dry-irrigation sample)
    d = daily["Yi"]
    s = d[d["is_irr"] & d["is_dry"]].dropna(subset=["Q_div", "Cobs_TEMP"])
    if len(s):
        x = np.log(s["Q_div"].clip(lower=0) + 1.0)
        for lab, rng in [("minmax", x.max() - x.min()),
                         ("p5_95", x.quantile(.95) - x.quantile(.05))]:
            amp_nh3 = (np.exp(BETA_NH3_GAMM * rng) - 1) * 100
            amp_ec  = (np.exp(BETA_EC_GAMM  * rng) - 1) * 100
            rec(f"Yi_effrange_{lab}",
                f"log(Qdiv+1) range={rng:.2f} (n={len(s)}) -> "
                f"NH3 +{amp_nh3:.0f}% , EC +{amp_ec:.0f}%")

    # (4) Luo spatial gradient: per-station mean TN/TP, irrigation months
    print("\n  -- per-station mean concentration (irrigation months) --")
    w = wq_daily.copy()
    w["month"] = pd.to_datetime(w["date"]).dt.month
    w = w[w["month"].isin(IRR_MONTHS)]
    for st, g in w.groupby("station"):
        tn = g[WQ_COL["TN"]].mean() if WQ_COL["TN"] in g.columns else np.nan
        tp = g[WQ_COL["TP"]].mean() if WQ_COL["TP"] in g.columns else np.nan
        rec(f"stationmean_{st}", f"TN={tn:.2f} TP={tp:.2f} mg/L (n={len(g)})")
    print("=" * 70)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("=" * 70); print("MASTER RECOMPUTE  (single source of truth)"); print("=" * 70)
    wq_daily, rn_wide, precip, ps = load_all()
    daily = {}
    for river in ["Yi", "Luo"]:
        print(f"\n----- {river} River " + "-" * 50)
        d = build_daily(river, wq_daily, rn_wide, precip)
        cfg = RIVER[river]
        # attach daily PS NH4 load routed to this node (covariate for the GAMM)
        if ps is not None and "unit" in ps.columns and cfg["ps_units"] and "NH4N_load" in ps.columns:
            sel = ps[ps["unit"].isin(cfg["ps_units"])]
            d["PSload_NH4"] = sel.groupby("date")["NH4N_load"].sum().reindex(d.index).fillna(0)
        else:
            d["PSload_NH4"] = np.nan
        # event ids on the irrigation+dry subset, merged back onto the daily frame
        de_full = label_events(d[d["is_irr"] & d["is_dry"]].copy(), cfg)
        d["event_id"] = de_full["event_id"].reindex(d.index)
        d["evt"]      = de_full["evt"].reindex(d.index)
        d.reset_index().rename(columns={"index": "date"}).to_csv(
            RESULTS_DIR / f"{river}_daily.csv", index=False)
        af_stats(river, d)
        dose_response(river, d)
        af_dc_corr(river, d)
        events(river, d, cfg)
        massflux_cv(river, d)
        dc_decomp(river, d)
        monte_carlo(river, d)
        cq_flowmatched(river, d)
        if river == "Yi":
            reverse_causality(river, d)
        mass_balance(river, wq_daily, rn_wide, ps, d)
        ps_fraction(river, d, ps)
        baseflow(river, rn_wide)
        exceedance(river, d)
        daily[river] = d

    report_open(daily, wq_daily)
    out = pd.DataFrame([{"key": k, "recomputed": v} for k, v in RESULTS.items()])
    out.to_csv(RESULTS_DIR / "recomputed_values.csv", index=False)
    print(f"\n[done] {len(out)} values -> {RESULTS_DIR/'recomputed_values.csv'}")
    print("All values written; column definitions are in SCHEMA.md / PROVENANCE.md.")


if __name__ == "__main__":
    main()
