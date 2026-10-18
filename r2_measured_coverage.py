import os, sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import pandas as pd
import master_recompute as mr
from r2_wq_measured import raw_path

MF = "restricted_for_review/model_frame_r2_full.csv"
YI_OUTLETS = ["YL10001", "YL10005", "YL10016", "YL10019"]

mf = pd.read_csv(MF)
mf = mf[mf["reach"] == "Yi"].copy()
mf["date"] = pd.to_datetime(mf["date"])
fin = lambda c: np.isfinite(pd.to_numeric(mf[c], errors="coerce"))
fx = mf[fin("NH3N_down") & fin("EC_down") & fin("Twater") & fin("API7")].copy()
rx = fx[fin("EC_up").reindex(fx.index)].copy()
print(f"model days: {len(fx)}, EC-ratio days: {len(rx)}")

raw = pd.read_csv(raw_path())
raw["date"] = pd.to_datetime(raw[["Year", "Month", "Day"]].rename(columns=str.lower))
st = raw["WaterQualityMonitoringStation"].astype(str).str.strip()
def raw_daily(station, det):
    s = raw.loc[st.str.lower() == station.lower()].groupby("date")[det].mean()
    return s[s.notna()]
print("\nmeasured values on the model days")
rows = []
for lab, frame, col, station, det in [
        ("EC_down  (absolute-EC response)", fx, "EC_down", "LongMenDaQiao", "ElectricConductance"),
        ("EC_down  (ratio numerator)",      rx, "EC_down", "LongMenDaQiao", "ElectricConductance"),
        ("EC_up    (ratio denominator)",    rx, "EC_up",   "TanTou",        "ElectricConductance"),
        ("Twater   (covariate)",            fx, "Twater",  "LongMenDaQiao", "WaterTemperature")]:
    rd = raw_daily(station, det)
    has = frame["date"].isin(rd.index)
    v_model = pd.to_numeric(frame.loc[has, col], errors="coerce").values
    v_raw = rd.reindex(frame.loc[has, "date"]).values
    diff = np.nanmax(np.abs(v_model - v_raw)) if has.any() else np.nan
    rows.append(dict(quantity=lab, model_days=len(frame), measured=int(has.sum()),
                     filled=int((~has).sum()), max_abs_diff_on_measured=round(float(diff), 3) if has.any() else np.nan))
t1 = pd.DataFrame(rows); print(t1.to_string(index=False))

w, rn, precip, ps = mr.load_all()
try:
    from r2_ps_qa import screen_ps
    ps, _ = screen_ps(ps, verbose=False)
except ImportError:
    pass
ps = ps.rename(columns={"unit": "N_unit"}) if "N_unit" not in ps.columns else ps
ps["date"] = pd.to_datetime(ps["date"])
print(f"\nNH4-N load records of the Yi outlets on the {len(fx)} model days")
rows = []
days = set(fx["date"])
for u in YI_OUTLETS:
    s = ps[(ps["N_unit"] == u) & ps["NH4N_load"].notna()]
    on = s[s["date"].isin(days)]
    rows.append(dict(outlet=u, days_with_record=on["date"].nunique(), share_of_model_days=f"{100*on['date'].nunique()/len(fx):.0f}%",
                     mean_load_when_reported_kgd=round(on["NH4N_load"].mean(), 2) if len(on) else np.nan,
                     full_period_mean_kgd=round(s["NH4N_load"].mean(), 2) if len(s) else np.nan))
t2 = pd.DataFrame(rows); print(t2.to_string(index=False))
rep = ps[ps["N_unit"].isin(YI_OUTLETS) & ps["NH4N_load"].notna() & ps["date"].isin(days)]
n_all4 = int((rep.groupby("date")["N_unit"].nunique() == 4).sum())
n_none = len(days - set(rep["date"]))
print(f"  all four reported: {n_all4} days; none reported: {n_none} days")
exp_share = {r.outlet: (r.full_period_mean_kgd or 0) for r in t2.itertuples()}
tot = sum(v for v in exp_share.values() if v == v)
if tot:
    miss = sum(exp_share[r.outlet] * (1 - r.days_with_record / len(fx)) for r in t2.itertuples() if exp_share[r.outlet] == exp_share[r.outlet])
    print(f"  load missing on the model days (full-period means): {100*miss/tot:.0f}%")
os.makedirs("r2_results", exist_ok=True)
t1.to_csv("r2_results/coverage_ec_model_days.csv", index=False, encoding="utf-8-sig")
t2.to_csv("r2_results/coverage_ps_model_days.csv", index=False, encoding="utf-8-sig")
print("\nsaved r2_results/coverage_ec_model_days.csv, r2_results/coverage_ps_model_days.csv")
