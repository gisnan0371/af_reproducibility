from __future__ import annotations
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import argparse
import itertools
import os
import numpy as np
import pandas as pd
from scipy import stats
import master_recompute as mr

from r2_wq_measured import build_daily_measured
OUTDIR = "r2_results"
IRR_MONTHS = getattr(mr, "IRR_MONTHS", [4, 5, 6, 7, 8, 9])
PRECIP_DRY = float(getattr(mr, "PRECIP_DRY", getattr(mr, "PRECIP_THR", 1.0)))
POLLUTANTS = {"NH3-N": "Cobs_NH3", "TN": "Cobs_TN", "TP": "Cobs_TP"}
MIN_WQ_DAYS = 2

GRID = dict(dr_high=[0.20, 0.25, 0.30], dr_low=[0.05, 0.10, 0.15],
            precip_thr=[0.5, 1.0, 2.0], min_dur=[2, 3, 5], lag=[0, 1])


def event_thresholds(cfg: dict) -> dict:
    src = cfg.get("event", cfg)
    return dict(q_div_high=float(src.get("q_div_high", 5.0)), dr_high=float(src.get("dr_high", 0.25)),
                q_div_low=float(src.get("q_div_low", 0.5)), dr_low=float(src.get("dr_low", 0.10)),
                min_dur=int(src.get("min_dur", src.get("min_duration", 3))))


def num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)


def precip_series(d: pd.DataFrame, precip) -> tuple[pd.Series, str]:
    cands = {}
    for c in ("precip", "precip_mm", "precip_sum", "P"):
        if c in d.columns:
            cands[f"daily.{c}"] = num(d, c)
    p = precip.copy() if precip is not None else None
    if p is not None:
        if not isinstance(p, pd.DataFrame):
            p = pd.DataFrame(p)
        if "date" in p.columns:
            p["date"] = pd.to_datetime(p["date"])
            st = [c for c in p.columns if "station" in c.lower()]
            numc = [c for c in p.columns if c != "date" and c not in st and pd.api.types.is_numeric_dtype(p[c])]
            if st and numc:
                cands["precip.long.sum"] = p.groupby("date")[numc[0]].sum()
                cands["precip.long.mean"] = p.groupby("date")[numc[0]].mean()
            elif numc:
                w = p.set_index("date")[numc]
                cands["precip.wide.sum"] = w.sum(axis=1)
                cands["precip.wide.mean"] = w.mean(axis=1)
                cands["precip.first"] = w.iloc[:, 0]
        else:
            w = p.select_dtypes("number")
            if not w.empty:
                cands["precip.idx.sum"] = w.sum(axis=1)
                cands["precip.idx.mean"] = w.mean(axis=1)
    best, best_agree = None, -1.0
    for k, s in cands.items():
        s = pd.to_numeric(s, errors="coerce").reindex(d.index).fillna(0.0)
        agree = float(((s < PRECIP_DRY) == d["is_dry"].astype(bool)).mean())
        if agree > best_agree:
            best, best_agree, best_key = s, agree, k
    if best is None:
        raise RuntimeError("No precipitation series found; cannot vary the dry-day threshold.")
    print(f"  precipitation source: {best_key}  (agreement with master is_dry at {PRECIP_DRY} mm: {best_agree:.1%})")
    if best_agree < 0.99:
        print("  warning: agreement below 99%")
    return best, best_key


def classify_days(d, thr, dr_high, dr_low, precip_thr, p_s):
    is_irr = d.index.month.isin(IRR_MONTHS)
    dry = (p_s < precip_thr).values
    q_div, dr = num(d, "Q_div").values, num(d, "DR").values
    ok = is_irr & dry & np.isfinite(q_div) & np.isfinite(dr)
    high = ok & (q_div >= thr["q_div_high"]) & (dr >= dr_high)
    low = ok & (q_div <= thr["q_div_low"]) & (dr <= dr_low)
    return np.where(high, "high", np.where(low, "low", "other"))


def strict_runs(labels: np.ndarray, index: pd.DatetimeIndex, min_dur: int):
    runs, i, n = [], 0, len(labels)
    while i < n:
        lab = labels[i]
        if lab == "other":
            i += 1; continue
        j = i
        # consecutive calendar days in the same class
        while j + 1 < n and labels[j + 1] == lab and (index[j + 1] - index[j]).days == 1:
            j += 1
        if j - i + 1 >= min_dur:
            runs.append((lab, i, j))
        i = j + 1
    return runs


def event_medians(d, runs, var_col, lag):
    c = num(d, var_col).shift(-lag)
    out = {"high": [], "low": []}
    for lab, i, j in runs:
        v = c.iloc[i:j + 1].dropna()
        if len(v) >= MIN_WQ_DAYS:
            out[lab].append(float(v.median()))
    return out


def mw(high, low):
    if len(high) >= 3 and len(low) >= 2:
        p = stats.mannwhitneyu(high, low, alternative="two-sided").pvalue
        mh, ml = np.median(high), np.median(low)
        pct = (mh - ml) / ml * 100 if ml != 0 else np.nan
        return p, mh - ml, pct
    return np.nan, np.nan, np.nan


def rng(s, nd=2):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.{nd}f} [{s.min():.{nd}f}\u2013{s.max():.{nd}f}]" if len(s) else "\u2014"


def legacy_events(d, cfg):
    de = mr.label_events(d[d["is_irr"].astype(bool) & d["is_dry"].astype(bool)].copy(), cfg)
    ev = de[de["event_id"] >= 0].copy()
    rows = []
    for eid, g in ev.groupby("event_id"):
        g = g.sort_index()
        span = (g.index.max() - g.index.min()).days + 1
        rows.append(dict(legacy_id=int(eid), Class=g["evt"].iloc[0], Start=g.index.min().date(),
                         End=g.index.max().date(), calendar_span_d=span, qualifying_dry_days=len(g),
                         wet_days_bridged=span - len(g),
                         n_days_with_NH3=int(num(g, "Cobs_NH3").notna().sum()),
                         Qdown_median=num(g, "Q_down").median(), Qdown_max=num(g, "Q_down").max(),
                         Qdiv_median=num(g, "Q_div").median(), DR_median=num(g, "DR").median()))
    t = pd.DataFrame(rows)
    t["__o"] = (t["Class"] != "high").astype(int)
    t = t.sort_values(["__o", "Start"]).drop(columns="__o").reset_index(drop=True)
    t.insert(0, "Event", [f"E{i+1}" for i in range(len(t))])
    return t


def strict_events_table(d, runs, high_qdown_p90):
    rows = []
    for k, (lab, i, j) in enumerate(runs):
        g = d.iloc[i:j + 1]
        note = []
        if lab == "low":
            note.append("canal closure")
        if num(g, "Q_down").max() > high_qdown_p90:
            note.append(f"includes day(s) with Q_down > P90 of high-diversion days ({high_qdown_p90:.0f} m\u00b3/s)")
        if num(g, "Cobs_NH3").notna().sum() < MIN_WQ_DAYS:
            note.append("no/insufficient downstream WQ")
        rows.append(dict(Class=lab, Start=g.index.min().date(), End=g.index.max().date(),
                         Duration_d=j - i + 1, n_NH3=int(num(g, "Cobs_NH3").notna().sum()),
                         n_TN=int(num(g, "Cobs_TN").notna().sum()), n_TP=int(num(g, "Cobs_TP").notna().sum()),
                         Qdown_med=num(g, "Q_down").median(), Qdown_min=num(g, "Q_down").min(), Qdown_max=num(g, "Q_down").max(),
                         Qdiv_med=num(g, "Q_div").median(), Qdiv_min=num(g, "Q_div").min(), Qdiv_max=num(g, "Q_div").max(),
                         DR_med=num(g, "DR").median(), DR_min=num(g, "DR").min(), DR_max=num(g, "DR").max(),
                         AF_med=num(g, "AF").median(),
                         NH3_med=num(g, "Cobs_NH3").median(), NH3_min=num(g, "Cobs_NH3").min(), NH3_max=num(g, "Cobs_NH3").max(),
                         TN_med=num(g, "Cobs_TN").median(), TN_min=num(g, "Cobs_TN").min(), TN_max=num(g, "Cobs_TN").max(),
                         TP_med=num(g, "Cobs_TP").median(), TP_min=num(g, "Cobs_TP").min(), TP_max=num(g, "Cobs_TP").max(),
                         Note="; ".join(note),
                         _fmt=dict(Qdown=rng(num(g, "Q_down"), 1), Qdiv=rng(num(g, "Q_div"), 1),
                                   DR=f"{num(g,'DR').median()*100:.0f} [{num(g,'DR').min()*100:.0f}\u2013{num(g,'DR').max()*100:.0f}]",
                                   NH3=rng(num(g, "Cobs_NH3")), TN=rng(num(g, "Cobs_TN")), TP=rng(num(g, "Cobs_TP")))))
    if not rows:
        print("  no events found")
        return pd.DataFrame(columns=["Event", "Class", "Start", "End"]), pd.DataFrame(columns=["Event", "Class"])
    t = pd.DataFrame(rows)
    t["__o"] = (t["Class"] != "high").astype(int)
    t = t.sort_values(["__o", "Start"]).drop(columns="__o").reset_index(drop=True)
    t.insert(0, "Event", [f"E{i+1}" for i in range(len(t))])
    fmt = pd.DataFrame({
        "Event": t["Event"], "Class": t["Class"], "Start": t["Start"], "End": t["End"],
        "Duration (d)": t["Duration_d"], "Days with WQ (NH\u2083-N/TN/TP)": [f"{a}/{b}/{c}" for a, b, c in zip(t.n_NH3, t.n_TN, t.n_TP)],
        "Q_down median [range] (m\u00b3/s)": [f["Qdown"] for f in t["_fmt"]],
        "Q_div median [range] (m\u00b3/s)": [f["Qdiv"] for f in t["_fmt"]],
        "DR median [range] (%)": [f["DR"] for f in t["_fmt"]],
        "NH\u2083-N median [range] (mg/L)": [f["NH3"] for f in t["_fmt"]],
        "TN median [range] (mg/L)": [f["TN"] for f in t["_fmt"]],
        "TP median [range] (mg/L)": [f["TP"] for f in t["_fmt"]],
        "Note": t["Note"]})
    return t.drop(columns="_fmt"), fmt


def spike_value(d, ccol, min_share=0.05):
    v = num(d, ccol).dropna()
    if v.empty:
        return np.nan
    vc = v.value_counts()
    return float(vc.index[0]) if len(vc) and vc.iloc[0] / len(v) >= min_share else np.nan


def flatrun_mask(d, ccol, min_run=3):
    v = num(d, ccol).dropna()
    m = pd.Series(False, index=d.index)
    if v.empty:
        return m
    grp = (v != v.shift()).cumsum()
    rl = v.groupby(grp).transform("size")
    m.loc[rl[rl >= min_run].index] = True
    return m


def event_tests(d, runs, lag=0, low_qdown_cap=None, drop_mask=None):
    rows = []
    for pname, ccol in POLLUTANTS.items():
        c = num(d, ccol).shift(-lag)
        if drop_mask is not None:
            c = c.where(~drop_mask.shift(-lag).fillna(False).values)
        hi, lo, loo_ok = [], [], []
        for lab, i, j in runs:
            g = d.iloc[i:j + 1]
            v = c.iloc[i:j + 1]
            if lab == "low" and low_qdown_cap is not None:
                v = v[num(g, "Q_down").values <= low_qdown_cap]
            v = v.dropna()
            if len(v) >= MIN_WQ_DAYS:
                (hi if lab == "high" else lo).append(float(v.median()))
        p, eff, pct = mw(hi, lo)
        signs = []
        pool = [("h", x) for x in hi] + [("l", x) for x in lo]
        for k in range(len(pool)):
            h2 = [x for m, (t_, x) in enumerate(pool) if t_ == "h" and m != k]
            l2 = [x for m, (t_, x) in enumerate(pool) if t_ == "l" and m != k]
            if len(h2) >= 1 and len(l2) >= 1:
                signs.append(np.sign(np.median(h2) - np.median(l2)))
        rows.append(dict(Pollutant=pname, n_high=len(hi), n_low=len(lo),
                         median_high=np.median(hi) if hi else np.nan, median_low=np.median(lo) if lo else np.nan,
                         Effect=eff, Change_pct=pct, MW_p=p,
                         LOO_all_positive=bool(signs) and all(s > 0 for s in signs)))
    return pd.DataFrame(rows)


def run_grid(d, thr, p_s):
    rows = []
    for dh, dl, pt, md, lag in itertools.product(GRID["dr_high"], GRID["dr_low"], GRID["precip_thr"],
                                                 GRID["min_dur"], GRID["lag"]):
        labels = classify_days(d, thr, dh, dl, pt, p_s)
        runs = strict_runs(labels, d.index, md)
        med = event_medians(d, runs, "Cobs_NH3", lag)
        p, eff, pct = mw(med["high"], med["low"])
        rows.append(dict(dr_high=dh, dr_low=dl, precip_thr=pt, min_dur=md, lag=lag,
                         n_high=len(med["high"]), n_low=len(med["low"]), Change_pct=pct, MW_p=p))
    g = pd.DataFrame(rows)
    valid = g.dropna(subset=["MW_p"])
    print(f"\ndefinition grid: {len(g)} combinations, {len(valid)} testable")
    if len(valid):
        print(f"  positive {(valid.Change_pct > 0).mean():.0%}, p < 0.05 {(valid.MW_p < 0.05).mean():.0%}, "
              f"median change {valid.Change_pct.median():+.1f}%")
    return g


def plot_grid(g, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    v = g.dropna(subset=["MW_p"]).copy()
    if v.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(v))
    colors = np.where(v.MW_p < 0.05, "#c62828", "#9e9e9e")
    ax.bar(x, v.Change_pct, color=colors, width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("definition (dr_high \u00d7 dr_low \u00d7 precip \u00d7 min_dur \u00d7 lag)")
    ax.set_ylabel("NH\u2083-N median change, high vs low (%)")
    ax.set_title(f"Strict events: positive in {(v.Change_pct>0).mean():.0%}, p<0.05 in {(v.MW_p<0.05).mean():.0%} "
                 f"of {len(v)} testable definitions", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--river", default="Yi", choices=["Yi", "Luo"])
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    wq_daily, rn_wide, precip, ps = mr.load_all()
    d = build_daily_measured(a.river, wq_daily, rn_wide, precip).sort_index()
    full_idx = pd.date_range(d.index.min(), d.index.max(), freq="D")
    if len(full_idx) != len(d):
        d = d.reindex(full_idx)
        d["is_irr"] = d.index.month.isin(IRR_MONTHS)
        d["is_dry"] = d["is_dry"].fillna(False)
    cfg = mr.RIVER[a.river]
    thr = event_thresholds(cfg)
    print(f"[{a.river}] day-level thresholds: {thr}")
    p_s, _ = precip_series(d, precip)

    leg = legacy_events(d, cfg)
    leg.to_csv(f"{OUTDIR}/events_legacy.csv", index=False, encoding="utf-8-sig")
    print(f"\nmaster label_events: {int((leg.Class=='high').sum())} high, {int((leg.Class=='low').sum())} low")
    print(leg.to_string(index=False))

    labels = classify_days(d, thr, thr["dr_high"], thr["dr_low"], PRECIP_DRY, p_s)
    runs = strict_runs(labels, d.index, thr["min_dur"])
    hi_idx = [np.arange(i, j + 1) for lab, i, j in runs if lab == "high"]
    high_days = d.iloc[np.concatenate(hi_idx)] if hi_idx else d.iloc[0:0]
    p90 = float(num(high_days, "Q_down").quantile(0.90)) if len(high_days) else np.inf
    st, fmt = strict_events_table(d, runs, p90)
    st.to_csv(f"{OUTDIR}/events_strict.csv", index=False, encoding="utf-8-sig")
    fmt.to_csv(f"{OUTDIR}/TableS8_events_formatted.csv", index=False, encoding="utf-8-sig")
    print(f"\nevents (min_dur={thr['min_dur']}): {int((st.Class=='high').sum())} high, "
          f"{int((st.Class=='low').sum())} low")
    print(fmt.to_string(index=False))

    cw = []
    for _, r in st.iterrows():
        inside = leg[(pd.to_datetime(leg.Start) <= pd.Timestamp(r.Start)) & (pd.to_datetime(leg.End) >= pd.Timestamp(r.End))]
        cw.append(dict(strict=r.Event, Class=r.Class, Start=r.Start, End=r.End,
                       legacy=inside.Event.iloc[0] if len(inside) else "(new / not in legacy)"))
    pd.DataFrame(cw).to_csv(f"{OUTDIR}/crosswalk_legacy_to_strict.csv", index=False, encoding="utf-8-sig")

    s4 = event_tests(d, runs, lag=0)
    s4.to_csv(f"{OUTDIR}/TableS4_strict.csv", index=False, encoding="utf-8-sig")
    print("\nevent contrast, lag 0")
    print(s4.to_string(index=False))
    s4b = event_tests(d, runs, lag=0, low_qdown_cap=p90)
    s4b.to_csv(f"{OUTDIR}/TableS4_strict_flowcomparable.csv", index=False, encoding="utf-8-sig")
    print(f"\nflow-comparable (low-event days with Q_down > {p90:.0f} m3/s dropped)")
    print(s4b.to_string(index=False))

    sv = spike_value(d, "Cobs_NH3")
    if np.isfinite(sv):
        spike = num(d, "Cobs_NH3").eq(sv).fillna(False)
        flat = flatrun_mask(d, "Cobs_NH3")
        n_obs = int(num(d, "Cobs_NH3").notna().sum())
        print(f"\nNH3-N = {sv:g} on {int(spike.sum())}/{n_obs} days ({100*spike.sum()/max(n_obs,1):.0f}%), "
              f"{int(flat.sum())} days in runs of >= 3 equal values")
        hi_days = np.concatenate([np.arange(i, j + 1) for lab, i, j in runs if lab == "high"] or [np.array([], int)])
        lo_days = np.concatenate([np.arange(i, j + 1) for lab, i, j in runs if lab == "low"] or [np.array([], int)])
        for nm, idxs in (("high-diversion event days", hi_days), ("low-diversion event days", lo_days)):
            if len(idxs):
                sub = num(d, "Cobs_NH3").iloc[idxs]
                obs = int(sub.notna().sum())
                print(f"   {nm}: {int(spike.iloc[idxs].sum())}/{obs} at {sv:g} "
                      f"({100*spike.iloc[idxs].sum()/max(obs,1):.0f}%)")
        variants = [("all days (as reported)", None),
                    (f"excluding days at {sv:g}", spike),
                    ("excluding days inside flat runs", flat),
                    (f"excluding days at {sv:g} OR in flat runs", spike | flat)]
        out = []
        for lbl, msk in variants:
            t = event_tests(d, runs, lag=0, drop_mask=msk)
            t.insert(0, "variant", lbl)
            out.append(t)
            r = t[t.Pollutant == "NH3-N"].iloc[0]
            print(f"   NH3-N | {lbl:42s} n_high={r.n_high:2.0f} n_low={r.n_low:2.0f} "
                  f"change={r.Change_pct:+6.1f}%  p={r.MW_p:.4f}")
        pd.concat(out).to_csv(f"{OUTDIR}/TableS4_censoring_variants.csv", index=False, encoding="utf-8-sig")
    else:
        print("\nno dominant NH3-N value")

    yrs = {}
    for lab, i, j in runs:
        n_wq = int(num(d.iloc[i:j + 1], "Cobs_NH3").notna().sum())
        if n_wq >= MIN_WQ_DAYS:
            yrs.setdefault(d.index[i].year, set()).add(lab)
    both = sorted([y for y, v in yrs.items() if {"high", "low"} <= v])
    print(f"\nyears with events: { {y: sorted(v) for y, v in sorted(yrs.items())} }")
    if both:
        sub = [(lab, i, j) for lab, i, j in runs if d.index[i].year in both]
        s4c = event_tests(d, sub, lag=0)
        s4c.insert(0, "years", ",".join(str(y) for y in both))
        s4c.to_csv(f"{OUTDIR}/TableS4_strict_sameyear.csv", index=False, encoding="utf-8-sig")
        print(f"same-year contrast {both}")
        print(s4c.to_string(index=False))
    else:
        print("no year has both high- and low-diversion events with water quality")

    g = run_grid(d, thr, p_s)
    g.to_csv(f"{OUTDIR}/FigS9_grid_strict.csv", index=False)
    parts = []
    for f, lab in (("TableS4_strict.csv", "main (lag 0)"), ("TableS4_strict_flowcomparable.csv", "flow-comparable"),
                   ("TableS4_strict_sameyear.csv", "within one year"), ("TableS4_censoring_variants.csv", None)):
        fp = f"{OUTDIR}/{f}"
        if os.path.isfile(fp):
            t = pd.read_csv(fp)
            if lab is not None:
                t.insert(0, "variant", lab)
            parts.append(t[["variant", "Pollutant", "n_high", "n_low", "median_high", "median_low", "Effect", "Change_pct", "MW_p"]])
    if parts:
        allc = pd.concat(parts, ignore_index=True).drop_duplicates()
        allc.to_csv(f"{OUTDIR}/event_contrasts_all.csv", index=False, encoding="utf-8-sig")
        print("\nall event contrasts")
        with pd.option_context("display.width", 200):
            print(allc.round({"median_high": 3, "median_low": 3, "Effect": 3, "Change_pct": 1, "MW_p": 4}).to_string(index=False))
    plot_grid(g, f"{OUTDIR}/FigS9_grid_strict.png")

    print(f"\nsaved to {OUTDIR}/")


if __name__ == "__main__":
    main()
