import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from pathlib import Path
import warnings, sys, os, argparse
warnings.filterwarnings('ignore')


_SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("AF_BASE_DIR", _SCRIPT_DIR))
INPUT_DIR = BASE_DIR / "input_data"
if not (INPUT_DIR / "input_water_quality_data.csv").exists():
    # fall back to the data folder that master_recompute points at
    try:
        import master_recompute as _mr
        _alt = Path(_mr.WQ_FILE).resolve().parent
        if (_alt / "input_water_quality_data.csv").exists():
            BASE_DIR, INPUT_DIR = _alt.parent, _alt
    except Exception:
        pass
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
SI_FIG_DIR = OUTPUT_DIR / "figures_SI"
R2_RESULTS_DIR = Path(os.environ.get("AF_R2_RESULTS", _SCRIPT_DIR / "r2_results"))
DPI = 600
PNG_DPI = 300
FORMATS = ["pdf", "png"]
SKIP_EXISTING = False
MIN_FONT = 8.0

FULL_START = "2018-09-01"
FULL_END   = "2021-12-31"
PS_START   = "2020-01-01"
IRR_MONTHS = [4, 5, 6, 7, 8, 9]
Q_MIN = 0.1
PRECIP_THR = 1.0
CONV = 86.4  # m3/s * mg/L -> kg/d

from matplotlib import font_manager as _fm
_AVAILABLE_FONTS = {f.name for f in _fm.fontManager.ttflist}
FONT = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans') if f in _AVAILABLE_FONTS), 'DejaVu Sans')
import functools as _ft
print = _ft.partial(print, flush=True)

plt.rcParams.update({
    'font.family': FONT,
    'font.size': 9,
    'mathtext.default': 'regular',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
})

C_YI  = '#C62828';  C_YI_L  = '#EF9A9A'
C_LUO = '#1565C0';  C_LUO_L = '#90CAF9'
C_IRR = '#FF8F00';  C_NON   = '#00695C'
C_PS  = '#D32F2F';  C_NPS   = '#388E3C'
C_LAT = '#F9A825';  C_UP    = '#1976D2'
C_BOX = '#A6CEE3'

POLLUTANTS = {
    'TN':   {'wq_col': 'TotalNitrogen',   'ps_load': 'TN_load',
             'unit': 'mg/L', 'std': 1.0,  'label': 'TN'},
    'TP':   {'wq_col': 'TotalPhosphorus',  'ps_load': 'TP_load',
             'unit': 'mg/L', 'std': 0.2,  'label': 'TP'},
    'NH3N': {'wq_col': 'AmmoniaNitrogen',  'ps_load': 'NH4N_load',
             'unit': 'mg/L', 'std': 1.0,  'label': r'NH$_3$-N'},
}

YI_CFG  = dict(div_cols=["MaoZhuangQu","LuHunGuangaiQu"],
               q_down="LongMenZhen", wq_up="TanTou", wq_dn="LongMenDaQiao",
               dr_high=0.25, dr_low=0.10, qdiv_high=5.0, qdiv_low=0.5,
               label="Yi River", color=C_YI, color_l=C_YI_L)
LUO_CFG = dict(div_cols=["LuShiLuoBeiQu","LuoNingLuoBeiQu","YiLuoQu","ZhongZhouQu"],
               q_down="BaiMaSi", wq_up="LingKou", wq_dn="BaiMaSi",
               dr_high=0.45, dr_low=0.20, qdiv_high=19.0, qdiv_low=8.0,
               label="Luo River", color=C_LUO, color_l=C_LUO_L)

FB_SEGMENTS = [
    ("Yi S1: Luanchuan-Tantou",   ["LuanChuan"], ["TanTou"],      ["BoYunLingQu", "YueJinQu"]),
    ("Yi S2: Tantou-Dongwan",     ["TanTou"],    ["DongWan"],     []),
    ("Yi S3: Dongwan-Luhun",      ["DongWan"],   ["LuHun"],       []),
    ("Yi S4: LuHun-LongMenZhen",  ["LuHun"],     ["LongMenZhen"], ["MaoZhuangQu", "LuHunGuangaiQu", "YiDongQu"]),
    ("Luo S1: LingKou-LuShi",     ["LingKou"],   ["LuShi"],       ["LuShiLuoBeiQu"]),
    ("Luo S2: LuShi-ChangShui",   ["LuShi"],     ["ChangShui"],   ["LuoNingLuoBeiQu"]),
    ("Luo S3: ChangShui-YiYang",  ["ChangShui"], ["YiYang"],      ["YiLuoQu"]),
    ("Luo S4: YiYang-BaiMaSi",    ["YiYang"],    ["BaiMaSi"],     ["ZhongZhouQu"]),
    ("YiLuo: Confluence-HSG",     ["LongMenZhen", "BaiMaSi"], ["HeiShiGuan"], []),
]
FB_LEGACY = [
    ("Yi S1: Luanchuan-Tantou",   ["LuanChuan"], ["TanTou"],      ["BoYunLingQu", "YueJinQu"]),
    ("Yi S2: Tantou-Dongwan",     ["TanTou"],    ["DongWan"],     []),
    ("Yi S3: Dongwan-Luhun",      ["DongWan"],   ["LuHun"],       []),
    ("Yi S4: LuHun-LongMenZhen",  ["LuHunQu"],   ["LongMenZhen"], ["MaoZhuangQu", "LuHunGuangaiQu"]),
    ("Luo S1: LingKou-LuShi",     ["LingKou"],   ["LuShi"],       ["LuShiLuoBeiQu"]),
    ("Luo S2: LuShi-ChangShui",   ["LuShi"],     ["ChangShui"],   ["LuShiLuoBeiQu", "LuoNingLuoBeiQu"]),
    ("Luo S3: ChangShui-YiYang",  ["ChangShui"], ["YiYang"],      ["LuoNingLuoBeiQu", "YiLuoQu"]),
    ("Luo S4: YiYang-BaiMaSi",    ["YiYang"],    ["BaiMaSi"],     ["YiLuoQu", "ZhongZhouQu"]),
    ("YiLuo: Confluence-HSG",     ["LongMenZhen", "BaiMaSi"], ["HeiShiGuan"], ["YiDongQu"]),
]
FLOOD_PCT = 0.98
PS_QA_REPORT = [None]

MB_SEGMENTS = {
    "Yi_lower": dict(label="Yi Lower",
        wq_up="TanTou", wq_dn="LongMenDaQiao", Q_up="LuHunQu", Q_dn="LongMenZhen",
        Q_div=["MaoZhuangQu","LuHunGuangaiQu"],
        ps_ids=["YL10001","YL10005","YL10016","YL10019"]),
    "Luo_seg2": dict(label="Luo Mid",
        wq_up="LuoNingChangShui", wq_dn="GaoYaZhai", Q_up="ChangShui", Q_dn="YiYang",
        Q_div=["YiLuoQu"],
        ps_ids=["YL10012","YL10013","YL10014"]),
    "Luo_seg3": dict(label="Luo Lower",
        wq_up="GaoYaZhai", wq_dn="BaiMaSi", Q_up="YiYang", Q_dn="BaiMaSi",
        Q_div=["ZhongZhouQu"],
        ps_ids=["YL10002","YL10003","YL10004","YL10006","YL10007","YL10015","YL10017","YL10018"]),
}


def _enforce_min_font(fig):
    for ax in fig.get_axes():
        items = list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + list(ax.texts) + \
                [ax.title, ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            items += list(leg.get_texts())
        for t in items:
            try:
                if t.get_fontsize() < MIN_FONT:
                    t.set_fontsize(MIN_FONT)
            except Exception:
                pass
    for t in fig.texts:
        if t.get_fontsize() < MIN_FONT:
            t.set_fontsize(MIN_FONT)


def _save(fig, folder, stem, tag=""):
    import time, os
    _enforce_min_font(fig)
    for fmt in FORMATS:
        out = folder / f"{stem}.{fmt}"
        if SKIP_EXISTING and out.exists():
            print(f"    .. {tag}{stem}.{fmt} exists, skipped"); continue
        t0 = time.time(); note = ""
        try:
            if fmt == "tiff":
                from PIL import Image
                hi = folder / f"{stem}.__600dpi__.png"
                fig.savefig(hi, format="png", dpi=DPI, bbox_inches='tight', pad_inches=0.05)
                im = Image.open(hi); im.load()
                im.convert("RGB").save(out, format="TIFF", compression=None, dpi=(DPI, DPI))
                os.remove(hi); note = ", uncompressed"
            elif fmt == "png":
                fig.savefig(out, format="png", dpi=PNG_DPI, bbox_inches='tight', pad_inches=0.05)
            else:
                fig.savefig(out, format=fmt, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
        except Exception as e:
            print(f"    !! {tag}{stem}.{fmt} failed: {e!r}"); continue
        mb = out.stat().st_size / 1e6 if out.exists() else 0
        print(f"    -> {tag}{stem}.{fmt}  ({mb:.1f} MB, {time.time() - t0:.1f} s{note})")
    plt.close(fig)


def savefig(fig, stem):
    _save(fig, FIG_DIR, stem)


def savefig_si(fig, stem):
    _save(fig, SI_FIG_DIR, stem, tag="SI: ")


def _sum_series(rn_wide, cols):
    s = pd.Series(0.0, index=rn_wide.index)
    for c in cols:
        if c in rn_wide.columns:
            s = s + rn_wide[c].fillna(0)
    return s


def _gauge_sum(rn_wide, cols):
    s = None
    for c in cols:
        if c not in rn_wide.columns:
            return pd.Series(np.nan, index=rn_wide.index)
        s = rn_wide[c].astype(float) if s is None else s + rn_wide[c].astype(float)
    return s if s is not None else pd.Series(0.0, index=rn_wide.index)


def river_stats(rn_wide, cfg):
    dates = rn_wide.index
    q_div = _sum_series(rn_wide, cfg['div_cols'])
    q_down = rn_wide[cfg['q_down']].fillna(0) if cfg['q_down'] in rn_wide.columns else pd.Series(0, index=dates)
    q_nat = q_down + q_div
    dr = q_div / q_nat.replace(0, np.nan)
    af = pd.Series(np.where(q_down > Q_MIN, q_nat / q_down, np.nan), index=dates)
    return pd.DataFrame({'Q_div': q_div, 'Q_down': q_down, 'Q_nat': q_nat, 'DR': dr, 'AF': af}, index=dates)


def load_all():
    print("loading data")
    start, end = pd.Timestamp(FULL_START), pd.Timestamp(FULL_END)

    wq = pd.read_csv(INPUT_DIR / "input_water_quality_data.csv")
    if 'pH' in wq.columns: wq = wq.rename(columns={'pH': 'Ph'})
    wq['datetime'] = pd.to_datetime(wq[['Year','Month','Day','Hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    wq['date'] = wq['datetime'].dt.normalize()
    wq = wq[(wq['date'] >= start) & (wq['date'] <= end)]
    wq_cols = [c for c in ['AmmoniaNitrogen','TotalPhosphorus','TotalNitrogen','DO','ElectricConductance','Turbidity','WaterTemperature'] if c in wq.columns]
    wq_daily = wq.groupby(['WaterQualityMonitoringStation','date'])[wq_cols].mean().reset_index()
    wq_daily.rename(columns={'WaterQualityMonitoringStation': 'station'}, inplace=True)
    actual = wq_daily['station'].unique()
    targets = ["LingKou","LuoNingChangShui","GaoYaZhai","BaiMaSi","TanTou","LongMenDaQiao","QiLiPu"]
    nm = {}
    for t in targets:
        for a in actual:
            if a.lower() == t.lower(): nm[a] = t
    wq_daily['station'] = wq_daily['station'].map(lambda x: nm.get(x, x))

    rn = pd.read_csv(INPUT_DIR / "input_runoff_data.csv")
    rn['date'] = pd.to_datetime(rn[['Year','Month','Day']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
    rn['Runoff'] = pd.to_numeric(rn['Runoff'], errors='coerce')
    rn = rn[(rn['date'] >= start) & (rn['date'] <= end)]
    rn_wide = rn.pivot_table(index='date', columns='RiverRunoffMonitoringStation', values='Runoff', aggfunc='mean')
    rn_wide = rn_wide.reindex(pd.date_range(start, end, freq='D'))
    rn_wide.index.name = 'date'

    met = pd.read_csv(INPUT_DIR / "input_metro_data.csv")
    met['datetime'] = pd.to_datetime(met[['Year','Month','Day','Hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    met['date'] = met['datetime'].dt.normalize()
    met = met[(met['date'] >= start) & (met['date'] <= end)]
    st_col = next((c for c in met.columns if 'station' in c.lower()), None)
    precip = met.groupby('date')['Precipitation'].sum().reset_index()
    precip.columns = ['date', 'precip_mm']
    met_d = met.drop_duplicates()
    psum_d = met_d.groupby('date')['Precipitation'].sum().reindex(precip['date']).values
    n_st = met_d.groupby('date')[st_col].nunique().reindex(precip['date']).values if st_col else 8
    precip['precip_mean_mm'] = psum_d / np.maximum(n_st, 1)
    dup = len(met) - len(met_d)
    if dup:
        print(f"  {dup} duplicate rows in the meteorology file (kept in precip_mm, as in master_recompute)")
    print(f"  meteorology: {met_d[st_col].nunique() if st_col else '?'} stations, "
          f"max daily basin mean {precip['precip_mean_mm'].max():.0f} mm, max daily sum {precip['precip_mm'].max():.0f} mm")

    ps = pd.read_csv(INPUT_DIR / "point_source" / "input_source.csv")
    if 'discharge(m3/day)' in ps.columns: ps = ps.rename(columns={'discharge(m3/day)': 'discharge'})
    ps['date'] = pd.to_datetime(ps['Date'])
    for c in ['TN_load','TP_load','NH4N_load','discharge']:
        if c in ps.columns: ps[c] = pd.to_numeric(ps[c], errors='coerce')
    ps = ps[(ps['date'] >= pd.Timestamp(PS_START)) & (ps['date'] <= end)]
    for c in ['TN', 'NH4N', 'TP']:
        if c in ps.columns: ps[c] = pd.to_numeric(ps[c], errors='coerce')
    ps = ps.rename(columns={'unit': 'N_unit'}) if 'N_unit' not in ps.columns and 'unit' in ps.columns else ps
    try:
        from r2_ps_qa import screen_ps
        ps, PS_QA_REPORT[0] = screen_ps(ps)
    except ImportError:
        print("  r2_ps_qa.py not found, point-source records not screened")

    print(f"  WQ stations: {sorted(wq_daily['station'].unique())}")
    print(f"  Hydro stations: {sorted(rn_wide.columns.tolist())}")
    print(f"  Period: {start.date()} to {end.date()}")
    return wq_daily, rn_wide, precip, ps


def fig2_flow_balance(rn_wide):
    print("  Fig. 2")
    sel = [FB_SEGMENTS[i] for i in [0, 3, 4, 6]]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    for idx, (name, qi, qo, qd) in enumerate(sel):
        ax = axes.flat[idx]
        missing = [c for c in (qi + qo + qd) if c not in rn_wide.columns]
        if missing:
            print(f"    {name}: {missing} not found, set to 0")
        dq = _gauge_sum(rn_wide, qi) - _gauge_sum(rn_wide, qo) - _sum_series(rn_wide, qd)
        data = [dq[dq.index.month == mn].dropna().values for mn in range(1, 13)]
        pos  = [mn for mn in range(1, 13) if len(data[mn - 1]) > 0]
        vals = [data[mn - 1] for mn in pos]
        if vals:
            bp = ax.boxplot(vals, positions=pos, widths=0.55, patch_artist=True,
                            showfliers=False, medianprops=dict(color='black', lw=1))
            for box in bp['boxes']:
                box.set_facecolor(C_BOX)
                box.set_edgecolor('gray'); box.set_linewidth(0.5)
        ax.axhline(0, color='black', lw=0.6)
        ax.axvspan(3.5, 9.5, alpha=0.10, color=C_IRR, lw=0)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylabel(r'$\Delta$Q (m$^3$/s)')
        ax.set_title(f'({chr(97+idx)}) {name}', fontsize=9, fontweight='bold', loc='left')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=7)
    fig.tight_layout()
    savefig(fig, "Fig2_flow_balance_audit")


def fig3_dr_timeseries(rn_wide):
    print("  Fig. 3")
    yi  = river_stats(rn_wide, YI_CFG)
    luo = river_stats(rn_wide, LUO_CFG)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    for ax, df, color, title in [(ax1, yi, C_YI, '(a) Yi River'), (ax2, luo, C_LUO, '(b) Luo River')]:
        ax.fill_between(df.index, 0, df['DR'].values * 100, alpha=0.5, color=color, lw=0)
        ax.set_ylabel('Diversion Ratio (%)')
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.set_ylim(0, 100)
        ax.axhline(50, color='gray', ls=':', lw=0.5)
        for y in [2019, 2020, 2021]:
            ax.axvspan(pd.Timestamp(f'{y}-04-01'), pd.Timestamp(f'{y}-09-30'), alpha=0.10, color=C_IRR, lw=0)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()
    savefig(fig, "Fig3_DR_timeseries")


def fig4_triple_panel(wq_daily, rn_wide, precip):
    print("  Fig. 4")
    dates = pd.date_range(FULL_START, FULL_END, freq='D')
    yi = river_stats(rn_wide, YI_CFG)
    pcol = 'precip_mean_mm' if 'precip_mean_mm' in precip.columns else 'precip_mm'
    p_s = precip.set_index('date')[pcol].reindex(dates).fillna(0)
    wq_dn = wq_daily[wq_daily['station'] == 'LongMenDaQiao'].set_index('date')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True,
                                          gridspec_kw={'height_ratios': [0.8, 1.3, 0.8]})
    ax1.bar(dates, p_s.values, color='#1E88E5', alpha=0.8, width=1, lw=0)
    ax1.set_ylabel('Basin-mean\nprecip. (mm d$^{-1}$)')
    ax1.set_ylim(p_s.max() * 1.15, 0)
    ax1.text(0.01, 0.92, '(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

    for pcol, lbl, c in [('TotalNitrogen', 'TN', C_YI), ('AmmoniaNitrogen', r'NH$_3$-N', '#FF6F00')]:
        if pcol in wq_dn.columns:
            ax2.plot(dates, wq_dn[pcol].reindex(dates), color=c, lw=0.5, alpha=0.8, label=lbl)
    ax2.axhline(1.0, color='red', ls=':', lw=0.8, alpha=0.7, label=r'NH$_3$-N Class III (1.0 mg/L)')
    ax2.set_ylabel('Concentration\n(mg/L)')
    ax2.legend(loc='upper right', framealpha=0.9, ncol=3, fontsize=8)
    ax2.text(0.01, 0.92, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top')

    ax3.fill_between(dates, 0, yi['DR'].reindex(dates).values * 100, alpha=0.5, color=C_YI, lw=0)
    ax3.set_ylabel('Diversion\nRatio (%)')
    ax3.set_ylim(0, 100)
    ax3.text(0.01, 0.92, '(c)', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top')

    for y in [2019, 2020, 2021]:
        for a in [ax1, ax2, ax3]:
            a.axvspan(pd.Timestamp(f'{y}-04-01'), pd.Timestamp(f'{y}-09-30'), alpha=0.10, color=C_IRR, lw=0)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    fig.tight_layout()
    savefig(fig, "Fig4_triple_panel_Yi")


def fig6_af_monthly(rn_wide):
    print("  Fig. 6")
    yi  = river_stats(rn_wide, YI_CFG)
    luo = river_stats(rn_wide, LUO_CFG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
    for ax, df, cfg, title in [(ax1, yi, YI_CFG, '(a) Yi River'), (ax2, luo, LUO_CFG, '(b) Luo River')]:
        data = [df.loc[df.index.month == m, 'AF'].dropna().values for m in range(1, 13)]
        bp = ax.boxplot(data, positions=range(1, 13), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', lw=1))
        for b in bp['boxes']:
            b.set_facecolor(cfg['color_l']); b.set_edgecolor(cfg['color']); b.set_lw(0.5); b.set_alpha(0.6)
        ax.axhline(1, color='black', lw=0.6); ax.axhline(2, color='red', ls=':', lw=0.6, alpha=0.5)
        ax.axvspan(3.5, 9.5, alpha=0.10, color=C_IRR, lw=0)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        ax.set_ylabel('Amplification Factor (AF)'); ax.set_title(title, fontweight='bold', loc='left')
        ax.set_ylim(0.5, min(12, np.nanpercentile(df['AF'].dropna(), 99) * 1.1))
    fig.tight_layout()
    savefig(fig, "Fig6_AF_monthly_boxplot")


def fig7_dry_window(wq_daily, rn_wide, precip):
    print("  Fig. 7")
    dates = rn_wide.index
    yi = river_stats(rn_wide, YI_CFG)
    p_s = precip.groupby('date')['precip_mm'].mean().reindex(dates)
    wq_dn = wq_daily[wq_daily['station'] == YI_CFG['wq_dn']].set_index('date')

    is_dry = (p_s < PRECIP_THR).astype(int)
    cum = np.zeros(len(is_dry))
    for i in range(len(is_dry)):
        cum[i] = (cum[i-1] + 1) if (is_dry.iloc[i] == 1 and i > 0) else is_dry.iloc[i]

    mask = (yi.index.month.isin(IRR_MONTHS) &
            (pd.Series(cum, index=dates) >= 3) &
            (yi['DR'] >= YI_CFG['dr_high']) &
            (yi['Q_div'] >= YI_CFG['qdiv_high']))

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]
        col = pcfg['wq_col']
        if col not in wq_dn.columns: continue
        c_obs = wq_dn[col].reindex(dates)
        c_cf  = c_obs / yi['AF']
        obs_v = c_obs[mask].dropna()
        cf_v  = c_cf.reindex(obs_v.index).dropna()
        common = obs_v.index.intersection(cf_v.index)
        ov, cv = obs_v[common].values, cf_v[common].values

        ax.scatter(cv, ov, alpha=0.5, s=12, color=C_YI, edgecolors='none', zorder=3)
        lim = [0, max(ov.max(), cv.max()) * 1.15]
        ax.plot(lim, lim, 'k--', lw=0.8, label='1:1 line')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel(f'Fixed-load scenario {pcfg["label"]}')
        ax.set_ylabel(f'Observed {pcfg["label"]}')
        if len(common) > 3:
            maf = yi.loc[common, 'AF'].mean()
            dc = (ov - cv).mean()
            ax.text(0.05, 0.95, f'n={len(common)}\nmean AF={maf:.2f}\n'
                    + r'mean $\Delta$C=' + f'{dc:+.2f}\n(algebraic; no test)',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))
        ax.set_title(f'({chr(97+j)}) {pcfg["label"]}', fontweight='bold', loc='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, "Fig7_dry_window_Yi")

def fig8_dose_response(wq_daily, rn_wide, precip):
    print("  Fig. 8")

    rn_wide = rn_wide.copy()
    rn_wide.index = pd.to_datetime(rn_wide.index).normalize()
    dates = rn_wide.index
    yi = river_stats(rn_wide, YI_CFG).reindex(dates)

    precip = precip.copy()
    precip["date"] = pd.to_datetime(precip["date"]).dt.normalize()
    p_s = precip.groupby("date", as_index=True)["precip_mm"].mean().reindex(dates)

    wq_daily = wq_daily.copy()
    wq_daily["date"] = pd.to_datetime(wq_daily["date"]).dt.normalize()
    wq_dn = (
        wq_daily[wq_daily["station"] == YI_CFG["wq_dn"]]
        .groupby("date", as_index=True)
        .mean(numeric_only=True)
        .reindex(dates)
    )

    dry_mask = p_s.reindex(dates).notna() & (p_s.reindex(dates) < PRECIP_THR)
    irr_mask = pd.Series(pd.Index(dates).month.isin(IRR_MONTHS), index=dates)
    mask_all = irr_mask & dry_mask & yi["AF"].reindex(dates).notna() & yi["DR"].reindex(dates).notna()
    mask = mask_all & pd.Series(dates >= pd.Timestamp(PS_START), index=dates)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]
        col = pcfg["wq_col"]

        if col not in wq_dn.columns:
            ax.set_visible(False)
            raise ValueError(f"Fig. 8: column {col} not found")

        c_dn = pd.to_numeric(wq_dn[col].reindex(dates), errors="coerce")
        dr = pd.to_numeric(yi["DR"].reindex(dates), errors="coerce")
        dr_pct = dr * 100.0
        dc = c_dn * dr

        data = (
            pd.concat([dr_pct.rename("DR"), dc.rename("DC")], axis=1)
            .loc[mask]
            .dropna()
        )

        if len(data) < 20:
            ax.set_visible(False)
            raise ValueError(f"Fig. 8 has insufficient data for {pname}: n={len(data)}")

        r_val = data["DR"].corr(data["DC"])

        data_all = pd.concat([dr_pct.rename("DR"), dc.rename("DC")], axis=1).loc[mask_all].dropna()
        print(f"    {pname}: n = {len(data)}, r = {r_val:.3f}; incl. 2019: n = {len(data_all)}, "
              f"r = {data_all['DR'].corr(data_all['DC']):.3f}")

        ax.scatter(
            data["DR"],
            data["DC"],
            alpha=0.2,
            s=8,
            color=C_YI,
            edgecolors="none",
            zorder=2,
        )

        bins = np.arange(0, 105, 10)
        bc = (bins[:-1] + bins[1:]) / 2
        bm, bse = [], []
        for s, e in zip(bins[:-1], bins[1:]):
            v = data.loc[(data["DR"] >= s) & (data["DR"] < e), "DC"]
            if len(v) >= 3:
                bm.append(v.mean())
                bse.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
            else:
                bm.append(np.nan)
                bse.append(np.nan)

        bm = np.array(bm)
        bse = np.array(bse)
        ok = ~np.isnan(bm)
        ax.errorbar(
            bc[ok],
            bm[ok],
            yerr=bse[ok],
            fmt="s-",
            color="black",
            markersize=4,
            lw=1.2,
            capsize=3,
            zorder=4,
        )

        z = np.polyfit(data["DR"], data["DC"], 1)
        xf = np.linspace(0, data["DR"].max(), 50)
        ax.plot(xf, np.polyval(z, xf), "k--", lw=0.8, alpha=0.5)

        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("Diversion Ratio (%)")
        ax.set_ylabel(r"$\Delta C$ (" + pcfg["unit"] + ")")
        ax.set_title(f"({chr(97 + j)}) {pcfg['label']}", fontweight="bold", loc="left", fontsize=9)
        ax.text(
            0.04,
            0.94,
            f"descriptive r={r_val:.3f}\nn={len(data)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )

    fig.tight_layout()
    savefig(fig, "Fig8_dose_response_Yi")

def fig9_luo_gradient(wq_daily, rn_wide):
    print("  Fig. 9")
    dates = rn_wide.index
    nodes_wq = [
        {"km": 63.0,  "stn": "LingKou"},
        {"km": 209.7, "stn": "LuoNingChangShui"},
        {"km": 314.7, "stn": "GaoYaZhai"},
        {"km": 340.7, "stn": "BaiMaSi"},
    ]
    div_order = ["LuShiLuoBeiQu","LuoNingLuoBeiQu","YiLuoQu","ZhongZhouQu"]
    div_kms = [150.5, 207.7, 289.7, 326.4]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]; ax2 = ax.twinx(); col = pcfg['wq_col']

        for is_irr, season, c_wq, c_dr, ls, mk in [
            (True,  'Irrigation',     C_YI,  C_YI_L,  '-', 'o'),
            (False, 'Non-irrigation', C_LUO, C_LUO_L, '--', 's')]:
            m_mask = dates.month.isin(IRR_MONTHS) if is_irr else ~dates.month.isin(IRR_MONTHS)
            sub_dates = dates[m_mask]

            x_wq, y_wq, y_err = [], [], []
            for node in nodes_wq:
                stn_data = wq_daily[(wq_daily['station'] == node['stn']) & (wq_daily['date'].isin(sub_dates))]
                if col in stn_data.columns:
                    vals = stn_data[col].dropna()
                    if len(vals) > 0:
                        x_wq.append(node['km']); y_wq.append(vals.mean())
                        y_err.append(1.96 * vals.std() / np.sqrt(len(vals)))
            if x_wq:
                ax.errorbar(x_wq, y_wq, yerr=y_err, fmt=f'{mk}{ls}', color=c_wq,
                            markersize=5, lw=1.3, capsize=2, label=f'{pcfg["label"]} {season}')

            q_bms = rn_wide.loc[sub_dates, 'BaiMaSi'].mean() if 'BaiMaSi' in rn_wide.columns else 1
            cum = 0; dr_x, dr_y = [63.0], [0]
            for dn, dk in zip(div_order, div_kms):
                if dn in rn_wide.columns:
                    cum += rn_wide.loc[sub_dates, dn].mean()
                dr_x.append(dk)
                dr_y.append(cum / (q_bms + cum) * 100 if (q_bms + cum) > 0 else 0)
            ax2.plot(dr_x, dr_y, 's:', color=c_dr, markersize=3, lw=0.8, alpha=0.7)

        for dk in div_kms:
            ax.axvline(dk, color=C_IRR, ls=':', lw=0.4, alpha=0.5)
        ax.axhline(pcfg['std'], color='red', ls=':', lw=0.5, alpha=0.3)
        ax.set_xlabel('Distance from source (km)')
        if j == 0: ax.set_ylabel('Concentration (mg/L)')
        ax2.set_ylim(0, 100)
        if j == 2: ax2.set_ylabel('Cumulative DR (%)')
        ax.set_title(f'({chr(97+j)}) {pcfg["label"]}', fontweight='bold', loc='left', fontsize=9)
        if j == 0: ax.legend(fontsize=5.5, loc='upper left')
    fig.tight_layout()
    savefig(fig, "Fig9_Luo_gradient")


def figS8_flow_matched_cq(wq_daily, rn_wide, precip):
    print("  Fig. S8")
    dates = rn_wide.index
    p_s = precip.groupby('date')['precip_mm'].mean().reindex(dates)
    wq_dn = wq_daily[wq_daily['station'] == YI_CFG['wq_dn']].set_index('date')
    if YI_CFG['q_down'] not in rn_wide.columns:
        print("    downstream gauge missing"); return
    q = rn_wide[YI_CFG['q_down']].reindex(dates)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]; col = pcfg['wq_col']
        if col not in wq_dn.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes); continue
        c = wq_dn[col].reindex(dates)
        d = pd.DataFrame({'C': c, 'Q': q, 'P': p_s}).dropna()
        d = d[(d['P'] < PRECIP_THR) & (d['Q'] > Q_MIN) & (d['C'] > 0)]
        d = d[d.index >= pd.Timestamp(PS_START)]
        d['irr'] = d.index.month.isin(IRR_MONTHS)
        irr, non = d[d['irr']], d[~d['irr']]
        if len(irr) < 10 or len(non) < 10:
            ax.text(0.5, 0.5, 'insufficient data', ha='center', transform=ax.transAxes); continue
        lo, hi = max(irr['Q'].min(), non['Q'].min()), min(irr['Q'].max(), non['Q'].max())
        print(f"    {pname}: Q_down overlap {lo:.1f}-{hi:.1f} m3/s, n = {len(irr)} / {len(non)}")
        irr, non = irr[(irr['Q'] >= lo) & (irr['Q'] <= hi)], non[(non['Q'] >= lo) & (non['Q'] <= hi)]
        fits = {}
        for sub, lbl, cl, mk in [(irr, 'Irrigation season', C_IRR, 'o'), (non, 'Non-irrigation', C_NON, 's')]:
            lx, ly = np.log10(sub['Q']), np.log10(sub['C'])
            sl, ic, r, p, se = stats.linregress(lx, ly)
            fits[lbl] = (sl, ic, len(sub))
            ax.scatter(lx, ly, s=9, alpha=0.35, color=cl, marker=mk, edgecolors='none', label=f'{lbl} (n={len(sub)})')
            xf = np.linspace(np.log10(lo), np.log10(hi), 50)
            ax.plot(xf, sl * xf + ic, color=cl, lw=1.4)
        x0 = np.log10(np.median(np.r_[irr['Q'].values, non['Q'].values]))
        def _shift(a, b):
            fa = np.polyfit(np.log10(a['Q']), np.log10(a['C']), 1)
            fb = np.polyfit(np.log10(b['Q']), np.log10(b['C']), 1)
            return np.polyval(fa, x0) - np.polyval(fb, x0)
        obs_shift = _shift(irr, non)
        pool = pd.concat([irr, non]); n_irr = len(irr)
        perm = []
        for _ in range(999):
            idx = rng.permutation(len(pool))
            perm.append(_shift(pool.iloc[idx[:n_irr]], pool.iloc[idx[n_irr:]]))
        p_perm = (np.sum(np.abs(perm) >= abs(obs_shift)) + 1) / (len(perm) + 1)
        pct = (10 ** obs_shift - 1) * 100
        print(f"    {pname}: intercept shift {pct:+.1f}%, permutation p = {p_perm:.3f}")
        ax.text(0.04, 0.05, f'intercept shift {pct:+.0f}%\n'
                            f'permutation p = {p_perm:.3f}\nQ range {lo:.1f}–{hi:.1f} m³/s',
                transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))
        ax.set_xlabel(r'$\log_{10}$($Q_{down}$) [m$^3$/s]')
        ax.set_ylabel(r'$\log_{10}$(' + pcfg['label'] + ') [mg/L]')
        ax.set_title(f'({chr(97+j)}) {pcfg["label"]}', fontweight='bold', loc='left', fontsize=9)
        if j == 0:
            ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    savefig_si(fig, "FigS8_flow_matched_CQ")


def fig10_exceedance(wq_daily, rn_wide):
    print("  Fig. 10")
    dates = rn_wide.index
    rivers = [('Yi River', YI_CFG, C_YI), ('Luo River', LUO_CFG, C_LUO)]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]
        for rname, rcfg, clr in rivers:
            ri = river_stats(rn_wide, rcfg)
            wq_stn = wq_daily[wq_daily['station'] == rcfg['wq_dn']].set_index('date')
            if pcfg['wq_col'] not in wq_stn.columns: continue
            data = pd.DataFrame({'DR': ri['DR'] * 100, 'C': wq_stn[pcfg['wq_col']].reindex(dates)}).dropna()
            data['exc'] = data['C'] > pcfg['std']
            bins = np.arange(0, 105, 10); bc = (bins[:-1] + bins[1:]) / 2
            ep = []
            for s, e in zip(bins[:-1], bins[1:]):
                v = data[(data['DR'] >= s) & (data['DR'] < e)]
                ep.append(v['exc'].mean() * 100 if len(v) >= 5 else np.nan)
            ep = np.array(ep); ok = ~np.isnan(ep)
            ax.plot(bc[ok], ep[ok], 'o-', color=clr, markersize=4, lw=1.2, label=rname)
        ax.axhline(50, color='gray', ls=':', lw=0.5)
        ax.set_xlabel('Diversion Ratio (%)'); ax.set_ylabel('Exceedance (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'({chr(97+j)}) {pcfg["label"]} > {pcfg["std"]} mg/L (Class III)', fontweight='bold', loc='left', fontsize=9)
        if j == 0: ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "Fig10_exceedance_probability")


def fig11_mc_uncertainty(rn_wide, precip, n_mc=10000, seed=42):
    print("  Fig. 11")
    dates = rn_wide.index
    p_s = precip.groupby('date')['precip_mm'].mean().reindex(dates)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
    for idx, (cfg, clr, lbl) in enumerate([(YI_CFG, C_YI, 'Yi River'), (LUO_CFG, C_LUO, 'Luo River')]):
        ax = axes[idx]
        q_div = _sum_series(rn_wide, cfg['div_cols'])
        q_down = rn_wide[cfg['q_down']].fillna(0) if cfg['q_down'] in rn_wide.columns else pd.Series(0.0, index=dates)
        mask = dates.month.isin(IRR_MONTHS) & (q_down > Q_MIN)
        qd, qn = q_down[mask].values, q_div[mask].values
        n = len(qd)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes); continue
        b_div = rng.uniform(-0.15, 0.15, n_mc)
        b_dwn = rng.uniform(-0.10, 0.10, n_mc)
        af_mean = np.empty(n_mc)
        for k in range(n_mc):
            e_div = 1 + b_div[k] + rng.uniform(-0.05, 0.05, n)
            e_dwn = 1 + b_dwn[k] + rng.uniform(-0.05, 0.05, n)
            qd_k, qn_k = qd * e_dwn, qn * e_div
            af_mean[k] = np.mean((qd_k + qn_k) / qd_k)
        ci = np.percentile(af_mean, [2.5, 50, 97.5])
        base = float(np.mean((qd + qn) / qd))
        half = (ci[2] - ci[0]) / (2 * ci[1]) * 100
        ax.hist(af_mean, bins=50, color=clr, alpha=0.55, edgecolor='white', lw=0.3)
        ax.axvline(ci[1], color='black', lw=1.2, label=f'median = {ci[1]:.2f}')
        ax.axvline(ci[0], color='black', ls='--', lw=0.8, label=f'95% CI {ci[0]:.2f}–{ci[2]:.2f}')
        ax.axvline(ci[2], color='black', ls='--', lw=0.8)
        ax.set_xlabel('Irrigation-season mean AF'); ax.set_ylabel('MC realisations')
        ax.set_title(f'({chr(97+idx)}) {lbl}: ±{half:.1f}% (measurement only)', fontweight='bold', loc='left', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        print(f"    {lbl}: n = {n}, AF = {base:.2f}, MC median {ci[1]:.2f}, 95% CI {ci[0]:.2f}-{ci[2]:.2f}, "
              f"half-width {half:.1f}%")
    fig.text(0.5, -0.02, 'Intervals reflect discharge-gauging (measurement) uncertainty only; '
                         'structural and scenario uncertainty are examined separately (Section 4.8).',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout()
    savefig(fig, "Fig11_MC_uncertainty")


def figS0_data_availability(wq_daily, rn_wide, precip, ps):
    print("  data availability")
    date_range = pd.date_range(FULL_START, FULL_END, freq='D')
    n_days = len(date_range)
    rows, labels = [], []

    for stn in sorted(wq_daily['station'].unique()):
        stn_dates = set(wq_daily.loc[wq_daily['station'] == stn, 'date'].values)
        avail = np.array([1 if d in stn_dates else 0 for d in date_range])
        rows.append(avail); labels.append(f'WQ: {stn}')

    for stn in sorted(rn_wide.columns):
        avail = (~rn_wide[stn].reindex(date_range).isna()).astype(int).values
        rows.append(avail); labels.append(f'Q: {stn}')

    for src in sorted(ps['N_unit'].unique()):
        src_dates = set(ps.loc[ps['N_unit'] == src, 'date'].values)
        avail = np.array([1 if np.datetime64(d) in src_dates else 0 for d in date_range])
        rows.append(avail); labels.append(f'PS: {src}')

    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.28)))
    cmap = plt.cm.colors.ListedColormap(['#f5f5f5', '#1976D2'])
    ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=5.5)
    month_starts = pd.date_range(FULL_START, FULL_END, freq='MS')
    xtick_pos = [(d - date_range[0]).days for d in month_starts]
    ax.set_xticks(xtick_pos[::3]); ax.set_xticklabels([d.strftime('%Y-%m') for d in month_starts][::3], rotation=45, fontsize=6)
    ax.set_title('Data Availability (blue = available, white = missing)', fontsize=11, fontweight='bold')
    for i, row in enumerate(matrix):
        mr = 1 - row.sum() / n_days
        clr = 'red' if mr > 0.3 else 'orange' if mr > 0.1 else 'green'
        ax.text(n_days + 5, i, f'{mr:.0%}', fontsize=5, va='center', color=clr)
    fig.tight_layout()
    savefig_si(fig, "Extra_data_availability")


def figS1_all_flow_balance(rn_wide):
    print("  Fig. S1")
    n = len(FB_SEGMENTS)
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.2))
    for idx, (name, qi, qo, qd) in enumerate(FB_SEGMENTS):
        ax = axes.flat[idx]
        missing = [c for c in (qi + qo + qd) if c not in rn_wide.columns]
        if missing:
            print(f"    {name}: {missing} not found, set to 0")
        dq = _gauge_sum(rn_wide, qi) - _gauge_sum(rn_wide, qo) - _sum_series(rn_wide, qd)
        data = [dq[dq.index.month == mn].dropna().values for mn in range(1, 13)]
        pos  = [mn for mn in range(1, 13) if len(data[mn - 1]) > 0]
        vals = [data[mn - 1] for mn in pos]
        if vals:
            bp = ax.boxplot(vals, positions=pos, widths=0.55, patch_artist=True,
                            showfliers=False, medianprops=dict(color='black', lw=0.8))
            for box in bp['boxes']:
                box.set_facecolor(C_BOX)
                box.set_edgecolor('gray'); box.set_linewidth(0.4)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvspan(3.5, 9.5, alpha=0.10, color=C_IRR, lw=0)
        ax.set_xlim(0.5, 12.5)
        ax.set_ylabel(r'$\Delta$Q (m$^3$/s)', fontsize=8)
        ax.set_title(f'({chr(97+idx)}) {name}', fontsize=8, fontweight='bold', loc='left')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
    fig.tight_layout()
    savefig_si(fig, "FigS1_flow_balance_all_segments")


def figS2_diversion_patterns(rn_wide):
    print("  Fig. S2")
    all_divs = [
        ("BoYunLingQu", "Yi: Boyunling"),    ("YueJinQu", "Yi: Yuejin"),
        ("MaoZhuangQu", "Yi: Maozhuang"),     ("LuHunGuangaiQu", "Yi: Luhun Irr."),
        ("LuShiLuoBeiQu", "Luo: Lushi LB"),   ("LuoNingLuoBeiQu", "Luo: Luoning LB"),
        ("YiLuoQu", "Luo: Yiluo"),           ("ZhongZhouQu", "Luo: Zhongzhou"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.9))
    for idx, (dname, label) in enumerate(all_divs):
        ax = axes.flat[idx]
        if dname not in rn_wide.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes); continue
        q = rn_wide[dname].astype(float)
        monthly = q.groupby(q.index.month).mean()
        ax.bar(range(1, 13), monthly.values, color=C_LUO if 'Luo' in label else C_YI, alpha=0.6)
        ax.axvspan(3.5, 9.5, alpha=0.10, color=C_IRR, lw=0)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
        ax.set_ylabel(r'Q$_{div}$ (m$^3$/s)', fontsize=8)
        ax.set_title(f'({chr(97+idx)}) {label}', fontsize=8, fontweight='bold', loc='left')
    fig.tight_layout()
    savefig_si(fig, "FigS2_diversion_channel_patterns")


def figS3_mass_balance(wq_daily, rn_wide, ps, precip=None):
    print("  Fig. S3")
    comps = [('up', 'Upstream\ninput'), ('point', 'Point\nsource'), ('lateral', 'Lateral\ninflow'), ('nps', 'Unclosed\nresidual')]
    segs = [('Yi_lower', 'Yi lower'), ('Luo_seg2', 'Luo mid'), ('Luo_seg3', 'Luo lower')]
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.4))
    dates = rn_wide.index
    for j, (sk, sl) in enumerate(segs):
        for i, pname in enumerate(POLLUTANTS):
            ax = axes[i, j]
            mb = _mb_components(wq_daily, rn_wide, ps, MB_SEGMENTS[sk], pname)
            mb = mb[dates >= pd.Timestamp(PS_START)]
            irr = mb.index.month.isin(IRR_MONTHS)
            vi = [mb.loc[irr, k].mean() for k, _ in comps]
            vn = [mb.loc[~irr, k].mean() for k, _ in comps]
            x = np.arange(len(comps)); w = 0.38
            ax.bar(x - w / 2, vn, w, color=C_NON, label='Non-irrigation')
            ax.bar(x + w / 2, vi, w, color=C_IRR, label='Irrigation')
            ax.axhline(0, color='black', lw=0.6)
            ax.set_xticks(x); ax.set_xticklabels([l for _, l in comps], fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{POLLUTANTS[pname]["label"]} load (kg/d)', fontsize=8)
            ax.set_title(f'({chr(97 + i * 3 + j)}) {sl}: {POLLUTANTS[pname]["label"]}', fontweight='bold', loc='left', fontsize=8.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='upper right')
            print(f"    {sl:9s} {pname:4s} irr: " + ", ".join(f"{k}={v:,.0f}" for (k, _), v in zip(comps, vi))
                  + " | non-irr: " + ", ".join(f"{k}={v:,.0f}" for (k, _), v in zip(comps, vn)))
    if precip is not None:
        mb = _mb_components(wq_daily, rn_wide, ps, MB_SEGMENTS['Yi_lower'], 'TN')
        p_s = precip.groupby('date')['precip_mm'].mean().reindex(dates)
        m = dates.month.isin(IRR_MONTHS) & (p_s < PRECIP_THR)
        d_ = mb.loc[m, ['div', 'up']].dropna()
        if len(d_):
            print(f"    Yi TN, dry irrigation days: diverted {d_['div'].mean():,.0f} of "
                  f"{d_['up'].mean():,.0f} kg/d ({100 * d_['div'].mean() / d_['up'].mean():.0f}%)")
    fig.tight_layout()
    savefig_si(fig, "FigS3_mass_balance_seasonal")


def figS4_luo_gradient_yearly(wq_daily, rn_wide):
    print("  Fig. S4")
    dates = rn_wide.index
    nodes_wq = [
        {"km": 63.0, "stn": "LingKou"}, {"km": 209.7, "stn": "LuoNingChangShui"},
        {"km": 314.7, "stn": "GaoYaZhai"}, {"km": 340.7, "stn": "BaiMaSi"},
    ]
    colors_year = {2019: '#E53935', 2020: '#FF9800', 2021: '#7B1FA2'}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    for j, (pname, pcfg) in enumerate(POLLUTANTS.items()):
        ax = axes[j]; col = pcfg['wq_col']
        for year, color in colors_year.items():
            sub_dates = dates[(dates.year == year) & dates.month.isin(IRR_MONTHS)]
            x_wq, y_wq = [], []
            for node in nodes_wq:
                stn_data = wq_daily[(wq_daily['station'] == node['stn']) & (wq_daily['date'].isin(sub_dates))]
                if col in stn_data.columns:
                    vals = stn_data[col].dropna()
                    if len(vals) > 0: x_wq.append(node['km']); y_wq.append(vals.mean())
            if x_wq: ax.plot(x_wq, y_wq, 'o-', color=color, markersize=5, lw=1.5, label=f'{year}')
        ax.axhline(pcfg['std'], color='red', ls=':', lw=0.5, alpha=0.3)
        ax.set_xlabel('Distance (km)'); ax.set_title(f'({chr(97+j)}) {pcfg["label"]}', fontweight='bold', loc='left', fontsize=9)
        if j == 0: ax.set_ylabel('Concentration (mg/L)'); ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_si(fig, "FigS4_Luo_gradient_yearly")


def fig12_management_zones(rn_wide, precip, zone_edges=(1.5, 2.0, 4.0)):
    print("  Fig. 12")
    dates = rn_wide.index
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.3))
    afs = {}
    for cfg, clr, lbl in [(YI_CFG, C_YI, 'Yi River'), (LUO_CFG, C_LUO, 'Luo River')]:
        ri = river_stats(rn_wide, cfg)
        af = ri['AF'][ri.index.month.isin(IRR_MONTHS)].dropna()
        afs[lbl] = (af, clr)
        xs = np.linspace(1, 12, 200)
        axes[0].plot(xs, [(af >= x).mean() * 100 for x in xs], color=clr, lw=1.5, label=lbl)
    for e in zone_edges:
        axes[0].axvline(e, color='gray', ls=':', lw=0.7)
    axes[0].set_xlabel('AF threshold x'); axes[0].set_ylabel('Irrigation-season days\nwith AF ≥ x (%)')
    axes[0].set_xlim(1, 12); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=8)
    axes[0].set_title('(a) AF exceedance frequency', fontweight='bold', loc='left', fontsize=9)
    xs = np.linspace(1, 12, 200)
    axes[1].plot(xs, (1 - 1 / xs) * 100, color='black', lw=1.5)
    for lbl, (af, clr) in afs.items():
        for q, mk in [(0.5, 'o'), (0.95, 's')]:
            v = af.quantile(q)
            axes[1].plot(v, (1 - 1 / v) * 100, marker=mk, ms=6, color=clr, ls='none',
                         label=f'{lbl} {"median" if q == 0.5 else "P95"} (AF {v:.1f})')
    axes[1].set_xlabel('AF'); axes[1].set_ylabel('1 − 1/AF (%): reduction of a fixed load\nentering below the offtake')
    axes[1].set_xlim(1, 12); axes[1].set_ylim(0, 100); axes[1].legend(fontsize=8, loc='lower right')
    axes[1].set_title('(b) 1 − 1/AF, fixed-load scenario', fontweight='bold', loc='left', fontsize=9)
    edges = [1.0] + list(zone_edges) + [np.inf]
    zc = ['#C8E6C9', '#FFF59D', '#FFCC80', '#EF9A9A']
    for i, (lbl, (af, clr)) in enumerate(afs.items()):
        bottom = 0
        for zi in range(4):
            sh = ((af >= edges[zi]) & (af < edges[zi + 1])).mean() * 100
            axes[2].bar(i, sh, bottom=bottom, color=zc[zi], edgecolor='white', width=0.6,
                        label=(f'AF {edges[zi]:g}–{edges[zi+1]:g}' if np.isfinite(edges[zi+1]) else f'AF > {edges[zi]:g}') if i == 0 else None)
            if sh > 6:
                axes[2].text(i, bottom + sh / 2, f'{sh:.0f}%', ha='center', va='center', fontsize=8)
            bottom += sh
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(list(afs.keys()))
    axes[2].set_ylabel('Irrigation-season days (%)'); axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    axes[2].set_title('(c) Share of days by AF interval', fontweight='bold', loc='left', fontsize=9)
    fig.tight_layout()
    savefig(fig, "Fig12_AF_management_zones")


def _mb_components(wq_daily, rn_wide, ps, scfg, pname, c_lat_factor=1.0, ps_ids=None):
    dates = rn_wide.index
    pcfg = POLLUTANTS[pname]; col = pcfg['wq_col']
    wq_up = wq_daily[wq_daily['station'] == scfg['wq_up']].set_index('date')
    wq_dn = wq_daily[wq_daily['station'] == scfg['wq_dn']].set_index('date')
    c_up = wq_up[col].reindex(dates) if col in wq_up.columns else pd.Series(np.nan, index=dates)
    c_dn = wq_dn[col].reindex(dates) if col in wq_dn.columns else pd.Series(np.nan, index=dates)
    q_up = rn_wide[scfg['Q_up']].reindex(dates) if scfg['Q_up'] in rn_wide.columns else pd.Series(np.nan, index=dates)
    q_dn = rn_wide[scfg['Q_dn']].reindex(dates) if scfg['Q_dn'] in rn_wide.columns else pd.Series(np.nan, index=dates)
    q_div = _sum_series(rn_wide, scfg['Q_div'])
    # LuHunQu is the release minus the two offtakes
    q_up_gross = q_up + q_div if scfg['Q_up'] == 'LuHunQu' else q_up
    q_lat = (q_dn - (q_up_gross - q_div)).clip(lower=0)
    q98 = q_dn.quantile(0.98)
    flood = q_dn > q98
    if not hasattr(_mb_components, "_reported"):
        _mb_components._reported = set()
    key = scfg['Q_dn']
    if key not in _mb_components._reported:
        _mb_components._reported.add(key)
        print(f"    [mass balance] {key}: flood days (Q > P98 = {q98:.0f} m3/s) excluded, {int(flood.sum())} days")
    ids = ps_ids if ps_ids is not None else scfg['ps_ids']
    sub = ps[ps['N_unit'].isin(ids)] if 'N_unit' in ps.columns else ps.iloc[0:0]
    l_ps = (sub.groupby('date')[pcfg['ps_load']].sum().reindex(dates)
            if pcfg['ps_load'] in sub.columns else pd.Series(np.nan, index=dates))
    l_ps[dates < pd.Timestamp(PS_START)] = np.nan
    l_ps = l_ps.where(dates < pd.Timestamp(PS_START), l_ps.fillna(0))
    up = c_up * q_up_gross * CONV
    div = c_up * q_div * CONV
    down = c_dn * q_dn * CONV
    lat = c_lat_factor * c_up * q_lat * CONV
    nps = down - up + div - l_ps - lat
    out = pd.DataFrame(dict(up=up, div=div, down=down, point=l_ps, lateral=lat, nps=nps,
                            q_up_gross=q_up_gross, q_div=q_div, q_dn=q_dn), index=dates)
    out.loc[flood, ['up', 'div', 'down', 'point', 'lateral', 'nps']] = np.nan
    return out


def figS5_lateral_sensitivity(wq_daily, rn_wide, ps):
    print("  Fig. S5")
    dates = rn_wide.index
    segs = [('Yi_lower', 'Yi lower'), ('Luo_seg3', 'Luo lower')]
    scen = [(0.0, 'clean ($C_{lat}$ = 0)', '#2E7D32', '-'),
            (1.0, 'base ($C_{lat}$ = $C_{up}$)', '#424242', '--'),
            (2.0, 'polluted ($C_{lat}$ = 2$C_{up}$)', '#C62828', ':')]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), sharex=True)
    tot, med, cover = {}, {}, {}
    for i, (sk, sl) in enumerate(segs):
        for j, pname in enumerate(POLLUTANTS):
            ax = axes[i, j]
            for f, lab, clr, ls in scen:
                mb = _mb_components(wq_daily, rn_wide, ps, MB_SEGMENTS[sk], pname, c_lat_factor=f)
                nps = mb['nps'][dates >= pd.Timestamp(PS_START)]
                mon = nps.resample('MS').mean()
                ax.plot(mon.index, mon.values, color=clr, ls=ls, lw=1.3, label=lab)
                tot[(sl, pname, f)] = nps.mean()
                med[(sl, pname, f)] = nps.median()
                if f == 1.0:
                    cover[(sl, pname)] = nps.notna().resample('MS').sum()
            ax.axhline(0, color='black', lw=0.6)
            for y in (2020, 2021):
                ax.axvspan(pd.Timestamp(f'{y}-04-01'), pd.Timestamp(f'{y}-09-30'), alpha=0.10, color=C_IRR, lw=0)
            ax.set_title(f'({chr(97 + i * 3 + j)}) {sl}: {POLLUTANTS[pname]["label"]}', fontweight='bold', loc='left', fontsize=8.5)
            if j == 0:
                ax.set_ylabel('Unclosed residual (kg/d)', fontsize=8)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    savefig_si(fig, "FigS5_lateral_inflow_sensitivity")
    print("    period-mean residual, kg/d (clean / base / polluted):")
    for sl in [s_ for _, s_ in segs]:
        for pname in POLLUTANTS:
            print(f"      {sl:9s} {pname:4s} " + " / ".join(f"{tot[(sl, pname, f)]:+,.0f}" for f, *_ in scen))
    rows = []
    for _, sl in segs:
        for pname in POLLUTANTS:
            for f, lab, *_ in scen:
                rows.append({'Segment': sl, 'Pollutant': POLLUTANTS[pname]['label'].replace('$', ''),
                             'Scenario': lab.replace('$', '').replace('_{lat}', 'lat').replace('_{up}', 'up'),
                             'residual_mean_kgd': f"{tot[(sl, pname, f)]:.0f}", 'residual_median_kgd': f"{med[(sl, pname, f)]:.0f}"})
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "TableS6_lateral_inflow_sensitivity.csv", index=False, encoding='utf-8-sig')
    print(f"    -> TableS6_lateral_inflow_sensitivity.csv ({len(rows)} rows)")
    for (sl, pname), c in cover.items():
        gaps = [d.strftime('%Y-%m') for d, n in c.items() if n == 0 and d >= pd.Timestamp(PS_START)]
        print(f"    coverage {sl:9s} {pname:4s}: months with no complete mass-balance day: {', '.join(gaps) if gaps else 'none'}")
    for sk, sl in segs:
        cfg = MB_SEGMENTS[sk]
        for pname in POLLUTANTS:
            col = POLLUTANTS[pname]['wq_col']
            for st in (cfg['wq_up'], cfg['wq_dn']):
                sub = wq_daily[wq_daily['station'] == st]
                if len(sub) and col in sub.columns:
                    ser = sub.set_index(pd.to_datetime(sub['date']))[col]
                    ser = ser[ser.index >= pd.Timestamp(PS_START)].reindex(pd.date_range(PS_START, dates.max()))
                    gaps = [d.strftime('%Y-%m') for d, n in ser.notna().resample('MS').sum().items() if n == 0]
                    print(f"    coverage {st:16s} {pname:4s}: months with no observation: {', '.join(gaps) if gaps else 'none'}")


def figS6_point_source_share(wq_daily, rn_wide, ps):
    print("  Fig. S6")
    dates = rn_wide.index
    scfg = MB_SEGMENTS['Yi_lower']
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0))
    for j, (pname, clr) in enumerate([('TN', C_YI), ('TP', C_NPS), ('NH3N', '#FF6F00')]):
        mb = _mb_components(wq_daily, rn_wide, ps, scfg, pname)
        mb = mb[dates >= pd.Timestamp(PS_START)]
        ratio = 100 * mb['point'] / mb['down'].replace(0, np.nan)
        mon = ratio.resample('MS').median()
        ax = axes[0, j]
        ax.plot(mon.index, mon.values, marker='o', ms=3, lw=1.2, color=clr)
        for y in (2020, 2021):
            ax.axvspan(pd.Timestamp(f'{y}-04-01'), pd.Timestamp(f'{y}-09-30'), alpha=0.10, color=C_IRR, lw=0)
        ax.set_ylim(bottom=0)
        ax.set_title(f'({chr(97 + j)}) {POLLUTANTS[pname]["label"]}', fontweight='bold', loc='left', fontsize=9)
        if j == 0:
            ax.set_ylabel('PS load / river load (%)', fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        non = ~mb.index.month.isin(IRR_MONTHS)
        d_ = mb.loc[non, ['point', 'down']].dropna()
        ax = axes[1, j]
        ax.scatter(d_['down'], d_['point'], s=7, alpha=0.4, color=clr, edgecolors='none')
        md = float((d_['point'] / d_['down']).mean()) if len(d_) else np.nan
        if len(d_):
            xs = np.array([0, d_['down'].max() * 1.02])
            ax.plot(xs, xs * md, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('River load (kg/d)', fontsize=8)
        if j == 0:
            ax.set_ylabel('Point-source load (kg/d)', fontsize=8)
        ax.set_title(f'({chr(100 + j)}) {POLLUTANTS[pname]["label"]}, Oct–Mar', fontweight='bold', loc='left', fontsize=9)
        r_avg = float((100 * d_['point'] / d_['down']).mean()) if len(d_) else np.nan
        ax.text(0.97, 0.05, f'mean daily ratio {100 * md:.1f}%', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
        print(f"    {pname}: Oct-Mar point-source/river load, mean of daily ratios {r_avg:.1f}% (n = {len(d_)})")
    fig.tight_layout()
    savefig_si(fig, "FigS6_point_source_share_Yi")


def figS7_reverse_causality(rn_wide):
    print("  Fig. S7")
    ri = river_stats(rn_wide, YI_CFG)
    d = ri[ri.index.month.isin(IRR_MONTHS)].dropna(subset=['Q_nat', 'Q_div', 'DR'])
    lag = ri['Q_nat'].shift(1).reindex(d.index)
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.5))
    pairs = [(d['Q_nat'], d['Q_div'], r'$Q_{nd}$ (m$^3$/s)', r'$Q_{div}$ (m$^3$/s)', '(a)'),
             (d['Q_nat'], 100 * d['DR'], r'$Q_{nd}$ (m$^3$/s)', 'DR (%)', '(b)'),
             (lag, d['Q_div'], r'$Q_{nd}$(t−1) (m$^3$/s)', r'$Q_{div}$(t) (m$^3$/s)', '(c)')]
    for ax, (x, y, xl, yl, tag) in zip(axes[:3], pairs):
        ok = x.notna() & y.notna() & (x > 0)
        rho = stats.spearmanr(x[ok], y[ok])[0]
        ax.scatter(x[ok], y[ok], s=5, alpha=0.35, color=C_YI, edgecolors='none')
        ax.set_xscale('log'); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.text(0.04, 0.96, f'ρ = {rho:.3f}\nn = {int(ok.sum())}', transform=ax.transAxes, fontsize=8, va='top')
        ax.set_title(tag, fontweight='bold', loc='left', fontsize=9)
        print(f"    {tag} Spearman rho = {rho:.3f} (n = {int(ok.sum())})")
    ax = axes[3]
    dq = ri['Q_div'].diff().reindex(d.index).dropna()
    ax.hist(dq, bins=40, color=C_YI, alpha=0.6, edgecolor='white', lw=0.3)
    share0 = 100 * (dq.abs() < 1e-9).mean()
    ax.set_xlabel(r'$\Delta Q_{div}$ (m$^3$/s d$^{-1}$)', fontsize=8); ax.set_ylabel('Days', fontsize=8)
    ax.text(0.96, 0.96, f'{share0:.0f}% of days\nunchanged', transform=ax.transAxes, fontsize=8, va='top', ha='right')
    ax.set_title('(d)', fontweight='bold', loc='left', fontsize=9)
    print(f"    (d) days with no change in Q_div: {share0:.0f}%")
    fig.tight_layout()
    savefig_si(fig, "FigS7_reverse_causality_Yi")


def figS9_threshold_grid():
    print("  Fig. S9")
    f = R2_RESULTS_DIR / "FigS9_grid_strict.csv"
    if not f.exists():
        print(f"    {f} not found, run r2_events_audit.py first"); return
    g = pd.read_csv(f).dropna(subset=['MW_p']).copy()
    g = g.sort_values('Change_pct').reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    ax.bar(np.arange(len(g)), g['Change_pct'], color='#78909C', width=1.0)
    ax.axhline(0, color='black', lw=0.7)
    med = g['Change_pct'].median(); q1, q3 = g['Change_pct'].quantile([0.25, 0.75])
    ax.axhline(med, color='black', lw=0.8, ls='--')
    ax.axhspan(q1, q3, color='black', alpha=0.08, lw=0)
    ax.set_xlabel(f'event definition (n = {len(g)}, sorted)'); ax.set_ylabel(r'NH$_3$-N median change, high vs low (%)')
    pos = (g['Change_pct'] > 0).mean() * 100
    ax.text(0.03, 0.95, f'positive: {pos:.0f}% of definitions\nmedian {med:+.1f}%, IQR {q1:+.1f} to {q3:+.1f}%',
            transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.9))
    ax.set_title('(a) All definitions, sorted by effect size', fontweight='bold', loc='left', fontsize=9)
    ax = axes[1]
    piv = g.pivot_table(index='dr_low', columns='dr_high', values='Change_pct', aggfunc='median')
    im = ax.imshow(piv.values, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{v:g}' for v in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f'{v:g}' for v in piv.index])
    ax.set_xlabel('DR threshold, high'); ax.set_ylabel('DR threshold, low')
    for (r, c), v in np.ndenumerate(piv.values):
        ax.text(c, r, f'{v:+.0f}%', ha='center', va='center', fontsize=8)
    ax.set_title('(b) Median change by DR thresholds', fontweight='bold', loc='left', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='median change (%)')
    fig.tight_layout()
    savefig_si(fig, "FigS9_strict_event_definition_grid")


def generate_tables(wq_daily, rn_wide, precip, ps):
    print("\ntables")

    pd.DataFrame([
        {'Data': 'Water quality', 'Stations': 7, 'Resolution': 'Hourly', 'Period': '2018.09-2021.12',
         'Parameters': 'TN, TP, NH3-N, DO, EC, Turbidity, T, pH'},
        {'Data': 'Hydrology', 'Stations': 21, 'Resolution': 'Daily', 'Period': '2019.01-2021.12',
         'Parameters': '11 river channel + 9 diversion channel discharge + 1 derived series (LuHunQu)'},
        {'Data': 'Meteorology', 'Stations': 8, 'Resolution': 'Hourly', 'Period': '2018.09-2021.12',
         'Parameters': 'Precipitation, Temperature, RH, Wind, Pressure'},
        {'Data': 'Point sources', 'Stations': 20, 'Resolution': 'Daily', 'Period': '2020.01-2021.12',
         'Parameters': 'Discharge, COD, NH4-N, TN, TP'},
    ]).to_csv(TABLE_DIR / "Table1_data_summary.csv", index=False)

    dates = rn_wide.index
    p_s = precip.groupby('date')['precip_mm'].mean().reindex(dates)
    result_rows = []
    for rname, rcfg in [('Yi River', YI_CFG), ('Luo River', LUO_CFG)]:
        ri = river_stats(rn_wide, rcfg)
        af = ri['AF'].dropna()
        af_irr = af[af.index.month.isin(IRR_MONTHS)]
        wq_dn = wq_daily[wq_daily['station'] == rcfg['wq_dn']].groupby('date').mean(numeric_only=True)
        mask_all = ri.index.month.isin(IRR_MONTHS) & p_s.notna() & (p_s < PRECIP_THR) & ri['AF'].notna() & ri['DR'].notna()
        mask = mask_all & (ri.index >= pd.Timestamp(PS_START))
        for pname, pcfg in POLLUTANTS.items():
            col = pcfg['wq_col']
            if col not in wq_dn.columns: continue
            c_dn = wq_dn[col].reindex(dates)
            dc = c_dn * (1 - 1 / ri['AF'])
            dr_pct = ri['DR'] * 100
            data = pd.DataFrame({'DR': dr_pct[mask], 'DC': dc[mask]}).dropna()
            r_val, p_val = stats.pearsonr(data['DR'], data['DC']) if len(data) > 10 else (np.nan, np.nan)
            d_all = pd.DataFrame({'DR': dr_pct[mask_all], 'DC': dc[mask_all]}).dropna()
            r_all = d_all['DR'].corr(d_all['DC']) if len(d_all) > 10 else np.nan
            result_rows.append({
                'River': rname, 'Pollutant': pcfg['label'].replace('$_3$', '3'),
                'AF_mean': f'{af.mean():.2f}', 'AF_median': f'{af.median():.2f}',
                'AF_P95': f'{af.quantile(0.95):.2f}',
                'AF_irr_mean': f'{af_irr.mean():.2f}',
                'DR_r': f'{r_val:.3f}', 'DR_p': f'{p_val:.1e}', 'DR_n': len(data),
                'DR_r_incl2019': f'{r_all:.3f}', 'DR_n_incl2019': len(d_all),
            })
    t2 = pd.DataFrame(result_rows)
    t2.to_csv(TABLE_DIR / "Table2_key_results.csv", index=False)
    print("  Table2_key_results.csv")
    for _, r in t2.iterrows():
        print(f"    {r['River']:9s} {r['Pollutant']:5s} AF {r['AF_irr_mean']}  r = {r['DR_r']} (n = {r['DR_n']}), "
              f"incl. 2019 r = {r['DR_r_incl2019']} (n = {r['DR_n_incl2019']})")

    hydro_info = {
        'HeiShiGuan': ('YiLuoRiver',1,18563), 'LuanChuan': ('YiRiver',1,340),
        'TanTou': ('YiRiver',1,1695), 'BoYunLingQu': ('YiRiver',2,0),
        'YueJinQu': ('YiRiver',2,0), 'DongWan': ('YiRiver',1,2623),
        'LuHun': ('YiRiver',1,3492), 'LuHunQu': ('YiRiver',3,3492),
        'MaoZhuangQu': ('YiRiver',2,0), 'LuHunGuangaiQu': ('YiRiver',2,0),
        'LongMenZhen': ('YiRiver',1,5318), 'YiDongQu': ('YiRiver',2,0),
        'LingKou': ('LuoRiver',1,2476), 'LuShi': ('LuoRiver',1,4623),
        'LuShiLuoBeiQu': ('LuoRiver',2,0), 'ChangShui': ('LuoRiver',1,6244),
        'LuoNingLuoBeiQu': ('LuoRiver',2,0), 'YiYang': ('LuoRiver',1,9713),
        'YiLuoQu': ('LuoRiver',2,0), 'BaiMaSi': ('LuoRiver',1,11891),
        'ZhongZhouQu': ('LuoRiver',2,0),
    }
    rows = []
    for name, (river, stype, area) in hydro_info.items():
        q = rn_wide[name].dropna() if name in rn_wide.columns else pd.Series(dtype=float)
        rows.append({
            'Station': name, 'River': river,
            'Type': {1: 'Channel', 2: 'Diversion', 3: 'Derived (release − offtakes)'}[stype],
            'Catchment_km2': area if area > 0 else '',
            'Q_mean_m3s': f'{q.mean():.2f}' if len(q) > 0 else '',
            'Q_max_m3s': f'{q.max():.1f}' if len(q) > 0 else '',
            'Data_coverage_pct': f'{(~q.isna()).mean()*100:.0f}%' if len(q) > 0 else '',
        })
    pd.DataFrame(rows).to_csv(TABLE_DIR / "TableS1_hydro_stations.csv", index=False)

    wq_rows = []
    for stn in sorted(wq_daily['station'].unique()):
        stn_data = wq_daily[wq_daily['station'] == stn]
        row = {'Station': stn, 'n_days': stn_data['date'].nunique()}
        for pname, pcfg in POLLUTANTS.items():
            col = pcfg['wq_col']
            if col in stn_data.columns:
                vals = stn_data[col].dropna()
                row[f'{pname}_mean'] = f'{vals.mean():.3f}'
                row[f'{pname}_std'] = f'{vals.std():.3f}'
                row[f'{pname}_median'] = f'{vals.median():.3f}'
        wq_rows.append(row)
    pd.DataFrame(wq_rows).to_csv(TABLE_DIR / "wq_station_statistics.csv", index=False)

    af_rows = []
    for rname, rcfg in [('Yi River', YI_CFG), ('Luo River', LUO_CFG)]:
        ri = river_stats(rn_wide, rcfg)
        af = ri['AF'].dropna()
        for period, m_list in [('Full period', list(range(1,13))),
                                ('Irrigation (Apr-Sep)', IRR_MONTHS),
                                ('Non-irrigation (Oct-Mar)', [10,11,12,1,2,3])]:
            vals = af[af.index.month.isin(m_list)]
            if len(vals) == 0: continue
            af_rows.append({
                'River': rname, 'Period': period, 'n': len(vals),
                'Mean': f'{vals.mean():.2f}', 'Median': f'{vals.median():.2f}',
                'P5': f'{vals.quantile(0.05):.2f}', 'P25': f'{vals.quantile(0.25):.2f}',
                'P75': f'{vals.quantile(0.75):.2f}', 'P95': f'{vals.quantile(0.95):.2f}',
                'Max': f'{vals.max():.1f}',
            })
    pd.DataFrame(af_rows).to_csv(TABLE_DIR / "TableS2_AF_statistics.csv", index=False)

    fb_rows = []
    print("  TableS3_flow_balance_audit.csv (downstream gauge <= P98)")
    for (name, qi, qo, qd), (_, lqi, lqo, lqd) in zip(FB_SEGMENTS, FB_LEGACY):
        q_in, q_out, q_dv = _gauge_sum(rn_wide, qi), _gauge_sum(rn_wide, qo), _sum_series(rn_wide, qd)
        l_in, l_dv = _gauge_sum(rn_wide, lqi), _sum_series(rn_wide, lqd)
        thr = q_out.quantile(FLOOD_PCT)
        keep = q_out <= thr
        dq = (q_in - q_out - q_dv)[keep].dropna()
        dq_leg = (l_in - q_out - l_dv)[keep].dropna()
        fb_rows.append({'Segment': name, 'n_days': len(dq), 'flood_threshold_m3s': f'{thr:.0f}',
                        'mean_deltaQ': f'{dq.mean():.2f}', 'median_deltaQ': f'{dq.median():.2f}',
                        'pct_negative': f'{(dq < 0).mean():.0%}', 'pct_positive': f'{(dq > 0).mean():.0%}',
                        'earlier_def_mean_deltaQ': f'{dq_leg.mean():.2f}',
                        'note': 'earlier definition: LuHunQu inflow, offtakes subtracted again' if 'Yi S4' in name else
                                ('earlier definition: canal also subtracted in the adjacent segment' if set(lqd) != set(qd) else '')})
        print(f"    {name:28s} n={len(dq):4d}  mean {dq.mean():7.2f}  median {dq.median():7.2f}  "
              f"neg {100*(dq<0).mean():5.1f}%   earlier definition {dq_leg.mean():7.2f}")
    pd.DataFrame(fb_rows).to_csv(TABLE_DIR / "TableS3_flow_balance_audit.csv", index=False, encoding='utf-8-sig')

    ps_rows = []
    reach_of = {}
    for sk, cfg in MB_SEGMENTS.items():
        for u in cfg['ps_ids']:
            reach_of[u] = cfg['label'] + ' (' + cfg['wq_dn'] + ')'
    for src in sorted(ps['N_unit'].unique()):
        sub = ps[ps['N_unit'] == src]
        row = {'Source_ID': src, 'Receiving_reach': reach_of.get(src, 'not used in the reach analyses'),
               'n_days': sub['date'].nunique()}
        rep = PS_QA_REPORT[0]
        if rep is not None and len(rep) and src in set(rep['N_unit']):
            rr = rep.set_index('N_unit').loc[src]
            row['n_days_reported'] = int(rr['n_days']); row['excluded_by_screen'] = int(rr['excluded'])
            row['share_excluded_pct'] = float(rr['share_excluded_pct'])
        for psl in ['TN_load', 'TP_load', 'NH4N_load']:
            if psl in sub.columns:
                row[f'n_valid_{psl.replace("_load", "")}'] = int(sub[psl].notna().sum())
        if 'discharge' in sub.columns:
            row['discharge_mean_m3d'] = f'{sub["discharge"].mean():.0f}'
        for psl in ['TN_load', 'TP_load', 'NH4N_load']:
            if psl in sub.columns:
                row[f'{psl}_mean_kgd'] = f'{sub[psl].mean():.2f}'
        ps_rows.append(row)
    pd.DataFrame(ps_rows).to_csv(TABLE_DIR / "TableS5_point_sources.csv", index=False)

    print(f"  tables in {TABLE_DIR}")


FIGURES = {}


def _register():
    FIGURES.update({
        'fig2':  fig2_flow_balance,
        'fig3':  fig3_dr_timeseries,
        'fig4':  fig4_triple_panel,
        'fig6':  fig6_af_monthly,
        'fig7':  fig7_dry_window,
        'fig8':  fig8_dose_response,
        'fig9':  fig9_luo_gradient,
        'fig10': fig10_exceedance,
        'fig11': fig11_mc_uncertainty,
        'fig12': fig12_management_zones,
        'S1': figS1_all_flow_balance,
        'S2': figS2_diversion_patterns,
        'S3': figS3_mass_balance,
        'S4': figS4_luo_gradient_yearly,
        'S5': figS5_lateral_sensitivity,
        'S6': figS6_point_source_share,
        'S7': figS7_reverse_causality,
        'S8': figS8_flow_matched_cq,
        'S9': figS9_threshold_grid,
        'extra': figS0_data_availability,
        'tables': generate_tables,
    })


def _call(fn, wq_daily, rn_wide, precip, ps):
    import inspect
    params = list(inspect.signature(fn).parameters)
    pool = dict(wq_daily=wq_daily, rn_wide=rn_wide, precip=precip, ps=ps)
    return fn(**{k: pool[k] for k in params if k in pool})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures and tables")
    ap.add_argument("--only", nargs="*", default=None,
                    help="figure keys, e.g. --only fig2 fig8 S9 (default: all except 'extra')")
    ap.add_argument("--formats", nargs="*", default=None, choices=["pdf", "tiff", "png"],
                    help="output formats (default: pdf png). TIFF is opt-in and written uncompressed.")
    ap.add_argument("--png-dpi", type=int, default=None, help="PNG resolution (default 300; use 600 for a submission raster)")
    ap.add_argument("--skip-existing", action="store_true", help="do not rewrite files that already exist")
    ap.add_argument("--dpi", type=int, default=None, help="raster dpi for TIFF (default 600)")
    a = ap.parse_args(argv)
    global FORMATS, SKIP_EXISTING, DPI, PNG_DPI
    if a.formats:
        FORMATS = list(a.formats)
    SKIP_EXISTING = a.skip_existing
    if a.dpi:
        DPI = a.dpi
    if a.png_dpi:
        PNG_DPI = a.png_dpi
    _register()
    print(f"formats: {', '.join(FORMATS)} (PNG {PNG_DPI} dpi" + (f", TIFF {DPI} dpi" if "tiff" in FORMATS else "") + ")")
    for dd in (FIG_DIR, SI_FIG_DIR, TABLE_DIR):
        dd.mkdir(parents=True, exist_ok=True)
    wq_daily, rn_wide, precip, ps = load_all()

    print(f"font: {FONT}")
    keys = a.only if a.only else [k for k in FIGURES if k != 'extra']
    failed = []
    import time
    for k in keys:
        if k not in FIGURES:
            print(f"  unknown key {k}"); continue
        t0 = time.time()
        try:
            _call(FIGURES[k], wq_daily, rn_wide, precip, ps)
            print(f"    ({k} done in {time.time() - t0:.0f} s)")
        except Exception as e:
            failed.append((k, repr(e)))
            print(f"  {k} failed: {e!r}")
    print(f"\nfigures: {FIG_DIR}\nSI figures: {SI_FIG_DIR}\ntables: {TABLE_DIR}")
    if failed:
        print("\nfailed:")
        for k, e in failed:
            print(f"  {k}: {e}")


if __name__ == "__main__":
    main()
