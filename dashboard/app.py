"""FSBI Insights: small-business data -> inflation forecasts, economic signals and trading strategies.

Reads only the files written by the notebook's last cell (Data/processed/dashboard/). Run:
    pip install streamlit plotly
    streamlit run app.py
A running dashboard reloads automatically on the next page refresh after the notebook is rerun.
Set FSBI_DASH_DATA (or use Settings > Advanced in the sidebar) if the data folder is somewhere else.
"""
import json
import os
import re
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="FSBI Insights", layout="wide", initial_sidebar_state="expanded")
HERE = Path(__file__).resolve().parent
_cands = [os.environ.get("FSBI_DASH_DATA"), os.environ.get("FSBI_DASH_DIR"), HERE / "Data" / "processed" / "dashboard", HERE.parent / "Data" / "processed" / "dashboard",
          HERE / "processed" / "dashboard", HERE / "data" / "processed" / "dashboard", HERE / "data" / "dashboard", HERE.parent / "data" / "processed" / "dashboard", HERE.parent / "processed" / "dashboard"]
DEFAULT = next((Path(p) for p in _cands if p and (Path(p) / "fsbi.csv").exists()), HERE / "Data" / "processed" / "dashboard")
_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}
Z = {80: 1.2816, 90: 1.6449, 95: 1.9600}
ncdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))

# ------------------------------------------------------------------ one visual language for every chart
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
ORANGE, NAVY, TEAL, PURPLE, AMBER, GREY, RED, GREEN = "#f26b1d", "#1f4e79", "#2a9d8f", "#7b5ea7", "#e9b949", "#9ca3af", "#c2410c", "#2f855a"
SERIES_COLOR = {"published": INK, "First print (history)": INK, "Online-selected (with FSBI)": ORANGE, "Online-selected (no FSBI)": "#f6ad7b", "Cleveland nowcast": NAVY, "Kalshi median": TEAL}
BLOCK_COLOR = {"own lags": GREY, "macro / market": NAVY, "FSBI national": ORANGE, "FSBI sector": "#f6ad7b", "FSBI region": AMBER, "Kalshi": TEAL}
pio.templates["fsbi"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK), title=dict(font=dict(size=15), x=0, xanchor="left", y=0.97),
    colorway=[ORANGE, NAVY, TEAL, PURPLE, AMBER, "#4c9be8", RED, GREY, "#8c564b", "#17becf"], paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="outside", tickcolor=GRID), yaxis=dict(gridcolor=GRID, zerolinecolor="#cbd2d9", linecolor=GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(size=12)),
    margin=dict(l=10, r=10, t=64, b=10), hoverlabel=dict(font_size=12), colorscale=dict(diverging="RdBu")))
pio.templates.default = "plotly_white+fsbi"
st.markdown("""<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
  h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
  h2 {font-size: 1.35rem !important; margin-top: 1.2rem !important;}
  h3 {font-size: 1.1rem !important;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #4b5563;}
  [data-testid="stMetricValue"] {font-size: 1.55rem;}
  .brand {font-size: 1.15rem; font-weight: 700; color: #1f2933; line-height: 1.3;}
  .brand span {display: block; font-size: 0.82rem; font-weight: 400; color: #6b7280;}
  .subtitle {color: #6b7280; font-size: 1.0rem; margin: 0.15rem 0 1.0rem 0;}
  .pill, .tag {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-right: 4px;}
  .pill-soon {background: #fff1e6; color: #c2410c;} .pill-later {background: #eef2f7; color: #1f4e79;} .pill-late {background: #fde8e8; color: #9b1c1c;}
  .tag-sig {background: #e6f4ec; color: #2f855a;} .tag-desc {background: #eef2f7; color: #1f4e79;} .tag-none {background: #f1f2f4; color: #6b7280;}
  .card-title {font-weight: 700; font-size: 1.05rem;} .small {color: #6b7280; font-size: 0.85rem;}
  .signal-row {padding: 6px 0 6px 10px; border-bottom: 1px solid #eef0f3; border-left: 3px solid transparent;}
  .selection {background: #f6f7f9; border: 1px solid #e5e7eb; border-radius: 6px; padding: 7px 12px; margin: 0.3rem 0 0.9rem 0; font-size: 0.88rem; color: #374151;}
  .selection b {color: #1f2933;} .selection .k {color: #6b7280;}
</style>""", unsafe_allow_html=True)

GLOSSARY = {
    "FSBI": "Fiserv Small Business Index: monthly card-spending sales, transaction counts and ticket (sales ÷ transactions) at small businesses, by sector, sub-sector and state.",
    "MoM / YoY": "Month-on-month and year-on-year percentage change.",
    "pp": "Percentage points: the difference between two percentages (0.30% vs 0.25% is 0.05 pp).",
    "Nowcast": "The Cleveland Fed's free, model-based forecast of this month's inflation; the benchmark to beat.",
    "Kalshi median": "The inflation print a prediction market prices as 50/50, read from its ladder of 'above K' contracts.",
    "RMSE ratio": "A model's typical error divided by the nowcast's. Below 1 means the model was more accurate.",
    "R²": "1 − (squared errors ÷ variance of the actuals). 1 is perfect; below 0 means worse than always predicting the average.",
    "Clark-West p": "Tests whether adding FSBI to a model genuinely improves its forecasts. Small (below 0.10) means yes.",
    "Sharpe ratio": "Average return per unit of risk, per year. Above 1 is strong; 0 means no reward for the risk.",
    "Drawdown": "Fall from the previous peak of a strategy's value.",
    "Adjusted p (Holm)": "The p-value after allowing for every strategy tried. Below 0.10 means unlikely to be luck.",
    "Design vs live period": "Models and strategies were chosen using months before the design end; later months are an honest live track record.",
    "Granger test": "Do past values of FSBI help predict a series beyond that series' own past? Run in both directions.",
    "IC": "Information coefficient: the rank correlation between a signal and the next period's returns across assets.",
}
PAGE_LABEL = {"1. FSBI explorer": "FSBI trends", "2. Inflation forecasts": "Inflation predictions", "3. Trading strategies": "Trading results", "4. Economic applications": "FSBI correlations"}
HIST, NOWCAST, KALSHI, SELECTED = "First print (history)", "Cleveland nowcast", "Kalshi median", "Online-selected (with FSBI)"
TARGETS = ["CPI", "core CPI", "PCE", "core PCE"]; KEY_OF = {"CPI": "cpi", "core CPI": "core_cpi", "PCE": "pce", "core PCE": "core_pce"}


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading the latest results...")
def load(folder, stamp):
    """`stamp` is the modification time of meta.json, so a new notebook run reloads the data on the next refresh."""
    p = Path(folder)
    def rd(n, dates=("date",)):
        f = p / n
        if not f.exists() or f.stat().st_size < 5:
            return pd.DataFrame()
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        for c in dates:
            if c in d:
                d[c] = pd.to_datetime(d[c])
        return d
    names = {"fsbi": "fsbi.csv", "inf": "inflation.csv", "strat": "strategies_monthly.csv", "trades": "strategies_trades.csv", "daily": "strategies_daily.csv", "lb": "leaderboard.csv",
             "sig": "signals.csv", "ssig": "strategy_signals.csv", "wret": "window_returns.csv", "xs": "sector_signals.csv", "port": "portfolios.csv", "regime": "regime.csv",
             "imp": "feature_importance.csv", "ladder": "ladder.csv"}
    d = {k: rd(v) for k, v in names.items()}
    for k, v in {"ablation": "ablation.csv", "anchor": "anchor_test.csv", "comp": "component_tests.csv", "regtest": "region_tests.csv", "risk": "risk.csv", "attrib": "attribution.csv",
                 "sens": "sensitivity.csv", "score": "scorecard.csv", "econ_nat": "econ_national.csv", "econ_sec": "econ_sector_sales.csv", "econ_state": "econ_state.csv",
                 "econ_map": "econ_state_map.csv", "leads": "econ_leads.csv", "lead_prof": "econ_lead_profiles.csv", "sc_ic": "smallcap_ic.csv",
                 "reversals": "reversals.csv", "macro_reads": "macro_reads.csv", "event": "fsbi_event.csv", "fi_grid": "fixed_income_grid.csv", "macro_ind": "macro_indicators.csv"}.items():
        d[k] = rd(v, dates=())
    d["meta"] = json.loads((p / "meta.json").read_text())
    if len(d["fsbi"]):
        d["fsbi"]["region"] = d["fsbi"]["region"].fillna("")
    return d


# ------------------------------------------------------------------ helpers
def with_measure(df, col, measure):
    df = df.sort_values("date").copy(); g = df.groupby(["level", "name", "region"], dropna=False)[col]
    df["value"] = (g.pct_change(fill_method=None) * 100 if measure == "MoM %" else g.pct_change(12, fill_method=None) * 100 if measure == "YoY %"
                   else ((1 + g.pct_change(3, fill_method=None)) ** 4 - 1) * 100 if measure.startswith("3-month") else df[col])
    return df


def metrics(r):
    r = pd.Series(r, dtype=float).dropna()
    if len(r) < 2 or r.std() == 0:
        return {"months": len(r)}
    down = r[r < 0]; c = (1 + r).cumprod()
    return {"months": len(r), "Sharpe": r.mean() / r.std() * np.sqrt(12), "Sortino": r.mean() / down.std() * np.sqrt(12) if len(down) > 1 and down.std() > 0 else np.nan,
            "win rate": (r[r != 0] > 0).mean() if (r != 0).any() else np.nan, "ROI %": 100 * (c.iloc[-1] - 1), "ann. return %": 100 * (c.iloc[-1] ** (12 / len(r)) - 1),
            "ann. vol %": 100 * r.std() * np.sqrt(12), "max drawdown %": 100 * (c / c.cummax() - 1).min(), "t-stat": r.mean() / r.std() * np.sqrt(len(r))}


kid = lambda name: re.sub(r"\W+", "_", name)[:48]
has = lambda df, n=1: df is not None and len(df) >= n
piv = lambda df, index, columns, values: df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
fnum = lambda v, f="{:.2f}", na="–": na if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def empty_note(msg="Nothing to show for this selection. Widen the timeline in the sidebar or change the filters."):
    st.caption(msg)


def header(title, subtitle):
    st.title(title); st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def selection(**items):
    """One line stating what the charts below are showing, built from the current widget values."""
    parts = " &nbsp;·&nbsp; ".join(f"<span class='k'>{k.replace('_', ' ')}:</span> <b>{v}</b>" for k, v in items.items() if v not in (None, "", []))
    st.markdown(f"<div class='selection'>Showing &nbsp; {parts}</div>", unsafe_allow_html=True)

def short_list(xs, n=4):
    xs = list(xs); return ", ".join(map(str, xs[:n])) + (f" +{len(xs) - n} more" if len(xs) > n else "")

def how_to_read(text, label="Method"):
    with st.expander(label):
        st.markdown(text)


_N = [0]
def show(fig, where=st, height=None, **layout):
    """Plot with an explicit key (two identical charts may be on screen at once) and the house layout."""
    fig.update_layout(**({"height": height} if height else {}), **{"legend_title_text": "", **layout})
    _N[0] += 1
    where.plotly_chart(fig, key=f"chart_{_N[0]}", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, **WIDE)


def time_axes(fig):
    fig.update_layout(hovermode="x unified"); fig.update_xaxes(title_text="")
    return fig


def xs_weights(row, frac, min_n=4, long_only=False):
    s = row.dropna()
    if len(s) < min_n:
        return {}
    n = max(1, int(round(len(s) * frac))); o = s.sort_values(kind="stable")
    if long_only:
        return {**{a: 1.0 / n for a in o.index[-n:]}, **{a: -1.0 / len(s) for a in s.index}} if n < len(s) else {}
    return {**{a: 0.5 / n for a in o.index[-n:]}, **{a: -0.5 / n for a in o.index[:n]}}


def verdict(row):
    """Plain-language verdict for a leaderboard row."""
    if pd.notna(row.get("Holm p")) and row["Holm p"] < 0.10:
        return "Significant"
    if pd.notna(row.get("bootstrap p")) and row["bootstrap p"] < 0.10 and row.get("Sharpe", 0) > 0:
        return "Promising"
    return "Inconclusive" if row.get("Sharpe", 0) > 0 else "Negative"


def evidence_dot(ev):
    """Tag: at least one result survives a significance test (placebo-only passes do not count), descriptive only, or none."""
    if "descriptive" in ev:
        return "<span class='tag tag-desc'>Descriptive</span>"
    counts = [int(m.group(1)) for m in re.finditer(r"(\d+)(?: of \d+)?(?: cells)? (?:significant|survive Holm|pass the reality check|with t >)", ev)]
    ps = [float(m.group(1)) for m in re.finditer(r"p = (\d*\.\d+)", ev)]
    return "<span class='tag tag-sig'>Significant</span>" if any(c > 0 for c in counts) or any(p < 0.10 for p in ps) else "<span class='tag tag-none'>Not significant</span>"


def scorecard(sc):
    """Readable scorecard: one block per row, text wraps instead of being cut off."""
    use_col = sc.columns[0]
    for _, r in sc.iterrows():
        ev = str(r.get("evidence", "")); dot = evidence_dot(ev)
        st.markdown(f"<div class='signal-row'><b>{r[use_col]}</b> {dot}<br>{r.get('result', '')}<br><span class='small'>Evidence: {ev}</span></div>", unsafe_allow_html=True)


asset_name = lambda a: a.replace("basket:", "").replace("spread:", "").replace("banks:", "banks ")


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("<div class='brand'>FSBI Insights<span>Fiserv Small Business Index: inflation, activity and markets</span></div>", unsafe_allow_html=True)
st.sidebar.write("")
page = st.sidebar.radio("Go to", list(PAGE_LABEL), format_func=PAGE_LABEL.get, key="page", label_visibility="collapsed")
st.sidebar.divider()
with st.sidebar.expander("Data folder"):
    folder = st.text_input("Folder written by the notebook", str(DEFAULT), key="folder")
if not (Path(folder) / "fsbi.csv").exists() or not (Path(folder) / "meta.json").exists():
    st.error(f"No dashboard data found in `{folder}`. Run the notebook through its last cell (section 7), or set the folder under **Advanced** in the sidebar.")
    st.stop()
D = load(folder, (Path(folder) / "meta.json").stat().st_mtime); meta = D["meta"]
TODAY = pd.Timestamp.now().normalize()
LIVE_M = pd.Timestamp(meta.get("live_month", meta["fsbi_last_month"])); DESIGN_END = pd.Timestamp(meta.get("design_end", meta.get("holdout_start")))
SCHED = meta.get("schedule", {}); PREV = pd.DataFrame(meta.get("previous", [])); CUR = pd.DataFrame(meta.get("current", [])); SPECS = meta.get("strategy_specs", {})


def days_to(date_str):
    return (pd.Timestamp(date_str) - TODAY).days if date_str else None


def countdown(date_str, fsbi=False):
    n = days_to(date_str)
    if n is None:
        return ""
    if n < 0:
        return f"{-n} day{'s' if n != -1 else ''} ago · " + ("add the new CSVs and rerun" if fsbi else "rerun to add the print")
    return "today" if n == 0 else f"in {n} day{'s' if n != 1 else ''}"


def pill(date_str, fsbi=False):
    n = days_to(date_str)
    cls = "pill-late" if n is not None and n < 0 else "pill-soon" if n is not None and n <= 7 else "pill-later"
    return f"<span class='pill {cls}'>{countdown(date_str, fsbi)}</span>"


def release_rows():
    order = ["fsbi", "cpi", "core_cpi", "pce", "core_pce", "retail_sales"]; out = []
    for key in [k for k in order if k in SCHED] + [k for k in SCHED if k not in order]:
        s_ = SCHED[key]
        n_ = days_to(s_["release date"])
        out.append({"Release": s_["series"], "For": pd.Timestamp(s_["next month"]).strftime("%b %Y"), "Date": pd.Timestamp(s_["release date"]).strftime("%a %d %b %Y"),
                    "Days left": n_, "When": countdown(s_['release date'], fsbi=key == 'fsbi'), "Source": s_.get("source", "")})
    return pd.DataFrame(out)


months = sorted(D["fsbi"].date.dt.strftime("%Y-%m").unique())
st.sidebar.markdown(f"**Latest FSBI month:** {LIVE_M:%B %Y}")
if SCHED:
    upcoming = sorted(((days_to(v["release date"]), v["series"], v["release date"]) for k, v in SCHED.items() if k not in ("core_cpi", "core_pce") and days_to(v["release date"]) is not None), key=lambda x: (x[0] < 0, abs(x[0])))
    if upcoming:
        n_, s_, d_ = upcoming[0]
        st.sidebar.markdown(f"**Next release:** {s_} {pill(d_, fsbi=s_.startswith('Fiserv'))}", unsafe_allow_html=True)
st.sidebar.divider()
start_m, end_m = st.sidebar.select_slider("Timeline", options=months, value=(months[max(0, len(months) - 61)], months[-1]), key="timeline",
                                          help="Limits every time chart and every statistic computed in the dashboard to these months.")
t0, t1 = pd.Timestamp(start_m), pd.Timestamp(end_m)
POSITION = st.sidebar.number_input("Position size ($)", value=10000, step=1000, min_value=0, key="position", help="Used to express strategy results in dollars.")
with st.sidebar.expander("Glossary"):
    for k_, v_ in GLOSSARY.items():
        st.markdown(f"**{k_}** · {v_}")
if meta.get("stale_downloads"):
    st.sidebar.warning(f"{len(meta['stale_downloads'])} downloads could not be refreshed on the last run and use older copies: {', '.join(meta['stale_downloads'][:6])}")
st.sidebar.caption(f"Notebook run {meta['updated']} · {meta.get('fsbi_file', 'FSBI file')} dated {meta.get('fsbi_file_modified', '?')} · "
                   f"design period ends {DESIGN_END:%b %Y}. Research tool, not investment advice.")


def shade_live(fig, end=None):
    """Mark the live track record (months after the design period) on a time chart."""
    fig.add_vrect(x0=DESIGN_END, x1=end or TODAY, fillcolor=PURPLE, opacity=0.06, line_width=0, annotation_text="live period", annotation_position="top left",
                  annotation_font=dict(size=11, color=PURPLE))
    return fig


def forecast_card(r, where, compact=False):
    """One target's live forecast: number, comparison with the benchmarks, release countdown and last month's miss."""
    sch = SCHED.get(KEY_OF.get(r.target, ""), {}); pv = PREV[PREV.target == r.target] if len(PREV) else pd.DataFrame()
    with where.container(border=True):
        status = str(r.get("status", ""))
        if sch and sch.get("next month") == r.month:
            rel = f"<span class='small'>release</span> {pd.Timestamp(sch['release date']):%a %d %b} {pill(sch['release date'])}"
        elif status.startswith("published"):
            rel = f"<span class='small'>already published: <b>{status.split()[-1]}%</b></span>"
        else:
            rel = f"<span class='small'>{status}</span>"
        st.markdown(f"<span class='card-title'>{r.target}</span><br>{rel}", unsafe_allow_html=True)
        st.metric(f"Forecast for {pd.Timestamp(r.month):%b %Y}, MoM", f"{r['forecast']:.2f}%", f"{r['forecast'] - r['Cleveland nowcast']:+.2f} pp vs nowcast", delta_color="off",
                  help=f"Selected model: {r.get('selected model', '')}. Nowcast {r['Cleveland nowcast']:.2f}%" + (f", Kalshi {r['Kalshi']:.2f}%" if pd.notna(r.get('Kalshi')) else "") + ".")
        if not compact and pd.notna(r.get("forecast YoY")):
            st.markdown(f"<span class='small'>Year on year</span> **{r['forecast YoY']:.2f}%**", unsafe_allow_html=True)
        if len(pv):
            p_ = pv.iloc[0]; miss = p_["miss (print - forecast)"]
            st.markdown(f"<span class='small'>Last month ({pd.Timestamp(p_['month']):%b}): printed <b>{p_['first print']:.2f}%</b> vs forecast {p_['forecast']:.2f}%, miss "
                        f"<b style='color:{RED if miss > 0 else NAVY}'>{miss:+.2f} pp</b></span>", unsafe_allow_html=True)


# ================================================================== 1. FSBI explorer
if page.startswith("1"):
    header("FSBI trends", f"Small-business sales, transactions and ticket through {LIVE_M:%B %Y}, by sector, sub-sector, region and state.")
    F = D["fsbi"]; METRIC = {"Sales": "sales_index", "Transactions": "txn_index", "Ticket (sales per transaction)": "ticket_index"}
    hc = st.columns(6); nat_ = F[F.level == "National"]
    for j, (lbl, col) in enumerate(METRIC.items()):
        for q, (ms_, unit) in enumerate((("YoY %", "year on year"), ("MoM %", "month on month"))):
            v_ = with_measure(nat_, col, ms_).dropna(subset=["value"])
            if len(v_) >= 2:
                hc[2 * j + q].metric(f"{lbl.split(' (')[0]}, {unit}", f"{v_.value.iloc[-1]:+.1f}%", f"{v_.value.iloc[-1] - v_.value.iloc[-2]:+.1f} pp", help="Change from the prior month's reading.")
    st.caption(f"National index, {LIVE_M:%B %Y}. Sales: card spending. Transactions: number of purchases. Ticket: sales per transaction.")
    MEASURE_LABEL = {"Index": "Index level", "MoM %": "Month on month %", "YoY %": "Year on year %", "3-month annualised %": "3-month annualised %"}
    regions = list(meta["regions"]); sectors = sorted(F.loc[F.level == "Sector", "name"].unique()); subsectors = sorted(F.loc[F.level == "Subsector", "name"].unique())
    levels = [l for l in ["National", "Sector", "Subsector", "Region", "State", "Region x sector"] if (F.level == l).any()]
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1.3, 2])
        level = c1.selectbox("Level", levels, index=levels.index("Sector") if "Sector" in levels else 0, key="p1_level")
        metric = c2.radio("Metric", list(METRIC), key="p1_metric", help="Sales = spending; transactions = volume; ticket = average purchase (price-like).")
        measure = c3.radio("Measure", list(MEASURE_LABEL), format_func=MEASURE_LABEL.get, index=2, horizontal=True, key="p1_measure")
        c1, c2, c3 = st.columns(3)
        sel_regions = c1.multiselect("Regions", regions, default=regions, key="p1_regions", disabled=level not in ("Region", "State", "Region x sector"))
        sel_sectors = c2.multiselect("Sectors", sectors, default=sectors[:5], key="p1_sectors", disabled=level not in ("Sector", "Region x sector"))
        sel_subs = c3.multiselect("Sub-sectors", subsectors, default=subsectors[:6], key="p1_subsectors", disabled=level != "Subsector")
    names_ = {"Sector": sel_sectors, "Subsector": sel_subs, "Region": sel_regions, "State": sel_regions, "Region x sector": sel_sectors}.get(level)
    selection(Level=level, Metric=metric.split(" (")[0], Measure=MEASURE_LABEL[measure], **({("Regions" if level in ("Region", "State") else "Series"): short_list(names_)} if names_ else {}), Period=f"{t0:%b %Y} to {t1:%b %Y}")
    tab_tr, tab_x, tab_reg = st.tabs(["Trends", "States and sectors", "Inflation regime"])
    with tab_tr:
        sub = F[F.level == level]
        if level in ("Sector", "Region x sector"):
            sub = sub[sub.name.isin(sel_sectors)]
        if level == "Subsector":
            sub = sub[sub.name.isin(sel_subs)]
        if level in ("Region", "State", "Region x sector"):
            sub = sub[sub.region.isin(sel_regions)]
        sub = with_measure(sub, METRIC[metric], measure); sub = sub[(sub.date >= t0) & (sub.date <= t1)].dropna(subset=["value"])
        sub["series"] = np.where(sub.level == "Region x sector", sub.region + " · " + sub.name, sub.name)
        if level == "State" and sub.series.nunique() > 12:
            top = sub.groupby("series").value.last().abs().sort_values(ascending=False).index[:12]; sub = sub[sub.series.isin(top)]
            st.caption("Showing the 12 states with the largest latest values; narrow the regions to see others.")
        if len(sub):
            fig = time_axes(px.line(sub, x="date", y="value", color="series", labels={"value": MEASURE_LABEL[measure]}, title=f"{metric}: {MEASURE_LABEL[measure].lower()}, by {level.lower()}"))
            if measure != "Index":
                fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
            show(fig, height=460)
        else:
            empty_note()
    with tab_x:
        c1, c2 = st.columns([1, 2])
        bm = c1.radio("Measure", ["YoY %", "MoM %"], format_func=MEASURE_LABEL.get, horizontal=True, key="p1_bmeasure")
        mlist = [m for m in months if pd.Timestamp(m) >= pd.Timestamp(months[0]) + pd.DateOffset(months=12 if bm == "YoY %" else 1)]
        pick = c2.select_slider("Month", options=mlist, value=mlist[-1], key=f"p1_month_{bm}")
        c1, c2 = st.columns([1.3, 1])
        stt = with_measure(F[F.level == "State"], METRIC[metric], bm); stt = stt[stt.date == pd.Timestamp(pick)].dropna(subset=["value"])
        if len(stt):
            lim = float(np.nanpercentile(np.abs(stt.value), 95)) or 1.0
            show(px.choropleth(stt, locations="name", locationmode="USA-states", color="value", scope="usa", color_continuous_scale="RdBu", range_color=(-lim, lim), hover_data={"region": True},
                               labels={"value": bm}, title=f"{metric.split(' (')[0]} by state, {MEASURE_LABEL[bm].lower()}, {pd.Timestamp(pick):%b %Y}"), c1, height=430)
        sec = with_measure(F[F.level == "Sector"], METRIC[metric], bm); sec = sec[sec.date == pd.Timestamp(pick)].dropna(subset=["value"]).sort_values("value")
        if len(sec):
            show(px.bar(sec, x="value", y="name", orientation="h", color=np.where(sec.value >= 0, "up", "down"), color_discrete_map={"up": TEAL, "down": RED}, labels={"value": bm, "name": ""},
                        title=f"Sectors ranked, {pd.Timestamp(pick):%b %Y}"), c2, height=430, showlegend=False)
        hm = with_measure(F[F.level == "Sector"], METRIC[metric], bm); hm = hm[(hm.date >= t0) & (hm.date <= t1)].dropna(subset=["value"])
        if len(hm):
            hm = hm.pivot_table(index="name", columns=hm.date.dt.strftime("%Y-%m"), values="value"); lim = float(np.nanpercentile(np.abs(hm.values), 95)) or 1.0
            show(px.imshow(hm, color_continuous_scale="RdBu", zmin=-lim, zmax=lim, aspect="auto", labels={"color": bm, "x": "", "y": ""}, title=f"{metric.split(' (')[0]} by sector and month ({MEASURE_LABEL[bm].lower()})"), height=420)
        else:
            empty_note("Year-on-year needs 12 months of history before the first month shown.")
    with tab_reg:
        R = D["regime"]
        r = R[(R.date >= t0) & (R.date <= t1)].dropna(subset=["demand_pull", "supply_squeeze"], how="all") if has(R) else pd.DataFrame()
        if len(r):
            fig = go.Figure([go.Scatter(x=r.date, y=100 * r.demand_pull, name="Demand-pull: price and volume both above trend", line=dict(color=TEAL, width=2.5)),
                             go.Scatter(x=r.date, y=100 * r.supply_squeeze, name="Supply-squeeze: price above trend, volume below", line=dict(color=ORANGE, width=2.5))])
            show(time_axes(fig), height=380, yaxis_title="share of sectors (%)", title="Which kind of inflation are small businesses seeing?")
            how_to_read("Each month, every FSBI sector is classified by whether its **ticket** (price-like) and **transactions** (volume) are above or below their own recent trend. "
                        "Price and volume rising together signals demand-driven inflation; price up with volume down signals supply-driven inflation (Shapiro, 2022). "
                        "Ticket is sales ÷ transactions, so noise in transaction counts biases readings toward supply-squeeze: compare the lines with their own history rather than with each other.")
        else:
            empty_note("No regime data in this export.")
    if meta.get("sector_map"):
        with st.expander("How FSBI sub-sectors map to ETFs and CPI components"):
            st.dataframe(pd.DataFrame(meta["sector_map"]), hide_index=True, **WIDE)

# ================================================================== 2. inflation forecasts
elif page.startswith("2"):
    header("Inflation predictions", "First-print CPI, core CPI, PCE and core PCE forecasts from FSBI, against the Cleveland Fed nowcast and Kalshi.")
    I = D["inf"]; oos = I[I.model != HIST]
    c1, c2, c3 = st.columns([2, 1, 1.4])
    target = c1.radio("Target", TARGETS, horizontal=True, key="p2_target"); freq2 = c2.radio("Measure", ["MoM", "YoY"], horizontal=True, key="p2_freq", help="Month on month or year on year.")
    fcol, acol = ("forecast", "actual") if freq2 == "MoM" else ("forecast_yoy", "actual_yoy")
    with c3.popover("Uncertainty settings"):
        unc_opts = sorted(oos.model.unique()); unc_default = next((m for m in (SELECTED, "Ridge") if m in unc_opts), unc_opts[0])
        unc_model = st.selectbox("Intervals use the past errors of", unc_opts, index=unc_opts.index(unc_default), key="p2_unc")
        level = st.select_slider("Interval width", options=[80, 90, 95], value=90, key="p2_level", format_func=lambda v: f"{v}%")
    zq = Z[level]
    def errors(tg, freq, model=None):
        g = oos[(oos.target == tg) & (oos.model == (model or unc_model))]
        return (g.actual - g.forecast).dropna() if freq == "MoM" else (g.actual_yoy - g.forecast_yoy).dropna()
    selection(Target=target, Measure="Month on month" if freq2 == "MoM" else "Year on year", Forecast_month=f"{LIVE_M:%b %Y}", Intervals=f"{level}% from past errors of {unc_model}", Period=f"{t0:%b %Y} to {t1:%b %Y}")
    tab_now, tab_hist, tab_drv, tab_fsbi = st.tabs(["Live forecast", "Track record", "Feature importance", "FSBI vs inflation"])
    with tab_now:
        if SCHED:
            with st.expander("Release calendar", expanded=True):
                rr = release_rows()
                st.dataframe(rr[["Release", "For", "Date", "Days left", "When"]], hide_index=True, key="release_table", **WIDE,
                             column_config={"Release": st.column_config.TextColumn(width="medium"), "Days left": st.column_config.NumberColumn(format="%d", width="small"), "When": st.column_config.TextColumn(width="large")})
        if len(CUR):
            cols = st.columns(len(CUR))
            for c, (_, r) in zip(cols, CUR.iterrows()):
                forecast_card(r, c)
            st.markdown(f"#### Will {target} come in above a given level?")
            row = CUR[CUR.target == target].iloc[0] if (CUR.target == target).any() else CUR.iloc[0]
            c1, c2 = st.columns([1, 3])
            freq = c1.radio("Measure", ["MoM", "YoY"], horizontal=True, key="ou_freq")
            point = row["forecast"] if freq == "MoM" else row.get("forecast YoY"); e = errors(row.target, freq)
            if point is None or pd.isna(point) or len(e) < 5:
                empty_note("Not enough past forecast errors for this target and measure.")
            else:
                sd = e.std(); thr = c1.number_input("Level K (%)", value=float(round(point, 1)), step=0.1, format="%.2f", key=f"ou_thr_{row.target}_{freq}")
                p_norm = 1 - float(ncdf((thr - point) / sd)); p_emp = float(((point + e.values) > thr).mean())
                c1.metric(f"Chance {row.target} {freq} > {thr:.2f}%", f"{p_norm:.0%}"); c1.caption(f"{p_emp:.0%} using the {len(e)} past forecast errors directly.")
                xs = np.linspace(point - 4 * sd, point + 4 * sd, 300); dens = np.exp(-0.5 * ((xs - point) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                fig = go.Figure(go.Scatter(x=xs, y=dens, line=dict(color=NAVY, width=2), name="forecast distribution", hoverinfo="skip"))
                if (xs > thr).any():
                    fig.add_trace(go.Scatter(x=xs[xs > thr], y=dens[xs > thr], fill="tozeroy", line=dict(width=0), fillcolor="rgba(242,107,29,0.40)", name=f"above {thr:.2f}%"))
                fig.add_vline(x=point, line_dash="dash", line_color=INK, annotation_text=f"forecast {point:.2f}%")
                with c2:
                    cc1, cc2 = st.columns([1.35, 1])
                    show(fig, cc1, height=300, title=f"{row.target} {freq}: range of likely outcomes", xaxis_title="%", yaxis_visible=False)
                    ks = np.round(np.arange(np.floor((point - 3 * sd) * 10) / 10, point + 3 * sd, 0.1), 1)
                    ladder = pd.DataFrame({"Above": ks, "Chance": 1 - ncdf((ks - point) / sd), "Past errors": [((point + e.values) > k).mean() for k in ks]})
                    cc2.dataframe(ladder, height=300, hide_index=True, **WIDE, column_config={"Above": st.column_config.NumberColumn(format="%.1f%%", width="small"),
                                  "Chance": st.column_config.ProgressColumn("Chance (model)", format="percent", min_value=0, max_value=1), "Past errors": st.column_config.NumberColumn("Using past errors", format="percent", width="small")})
                st.caption("The ladder matches Kalshi's 'above K' contracts: compare it with market prices to see where the model and the market disagree.")
            if len(PREV):
                with st.expander("Last month: every forecast against the first print"):
                    st.dataframe(PREV.set_index("target").round(3), **WIDE)
        else:
            empty_note("No live forecast in this export.")
    with tab_hist:
        c1, c2 = st.columns([1.3, 2])
        per = c1.radio("Score which months?", ["Design period", "Live track record", "All"], horizontal=True, key="p2_period",
                       help=f"Design period: months before {DESIGN_END:%b %Y}, used to choose the models. Live: real-time forecasts since then, never used for any choice.")
        so = oos if per == "All" or "period" not in oos else oos[oos.period == ("design" if per == "Design period" else "live")]
        sc_rows = {}; ref_ = so[(so.target == target) & (so.model == NOWCAST)]; e0 = (ref_[acol] - ref_[fcol]).dropna()
        for m, g in so[so.target == target].groupby("model"):
            e_ = (g[acol] - g[fcol]).dropna()
            if len(e_) < 5:
                continue
            a_ = g.loc[e_.index, acol]; sst = ((a_ - a_.mean()) ** 2).sum()
            sc_rows[m] = {"months": len(e_), "RMSE": np.sqrt((e_ ** 2).mean()), "MAE": e_.abs().mean(), "R2": 1 - (e_ ** 2).sum() / sst if sst > 0 else np.nan,
                          "RMSE / nowcast": np.sqrt((e_ ** 2).mean()) / np.sqrt((e0 ** 2).mean()) if len(e0) else np.nan, "hits 0.1 bucket": (g.loc[e_.index, fcol].round(1) == a_.round(1)).mean()}
        sc = pd.DataFrame(sc_rows).T.sort_values("RMSE") if sc_rows else pd.DataFrame()
        avail = sorted(oos[oos.target == target].model.unique())
        default = [m for m in [SELECTED, NOWCAST, KALSHI] if m in avail]
        chosen = c2.multiselect("Forecasts to show (the first gets an uncertainty band)", avail, default=default, key="p2_models")
        d = oos[(oos.target == target) & oos.model.isin(chosen) & (oos.date >= t0) & (oos.date <= t1)].dropna(subset=[fcol])
        hist = I[(I.target == target) & (I.model == HIST) & (I.date >= t0) & (I.date <= t1)].dropna(subset=[acol])
        if not st.toggle("Show the full history of published prints", value=False, key="p2_hist") and len(d):
            hist = hist[hist.date.between(d.date.min(), d.date.max())]
        if len(hist) or len(d):
            fig = go.Figure()
            if chosen and len(d):
                g0 = d[d.model == chosen[0]].sort_values("date"); a0 = oos[(oos.target == target) & (oos.model == chosen[0])]; s0 = (a0[acol] - a0[fcol]).std()
                if len(g0) and np.isfinite(s0):
                    fig.add_trace(go.Scatter(x=np.r_[g0.date, g0.date[::-1]], y=np.r_[g0[fcol] + zq * s0, (g0[fcol] - zq * s0)[::-1]], fill="toself", fillcolor="rgba(242,107,29,0.12)",
                                             line=dict(width=0), name=f"{level}% band", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist.date, y=hist[acol], mode="lines+markers", name="Published (first print)", line=dict(color=INK, width=2.5), marker=dict(size=5)))
            for m, g in d.groupby("model"):
                fig.add_trace(go.Scatter(x=g.date, y=g[fcol], mode="lines+markers", name=m, marker=dict(size=4), line=dict(color=SERIES_COLOR.get(m), dash="dot" if m in (NOWCAST, KALSHI) else "solid", width=2)))
            shade_live(fig, end=max(d.date.max() if len(d) else t1, hist.date.max() if len(hist) else t1) + pd.offsets.MonthEnd(0))
            show(time_axes(fig), height=440, title=f"{target} {freq2}: published vs forecast (the last point is the live forecast)", yaxis_title="%")
        else:
            empty_note()
        if len(sc):
            c1, c2 = st.columns([1.35, 1])
            c1.dataframe(sc.reset_index().rename(columns={"index": "model"}), height=380, hide_index=True, **WIDE, column_config={
                "RMSE": st.column_config.NumberColumn("Typical error (RMSE, pp)", format="%.3f"), "MAE": st.column_config.NumberColumn("MAE (pp)", format="%.3f"),
                "R2": st.column_config.NumberColumn("R²", format="%.2f"), "RMSE / nowcast": st.column_config.NumberColumn("Error vs nowcast", format="%.2f", help="Below 1 = more accurate than the nowcast."),
                "hits 0.1 bucket": st.column_config.NumberColumn("Exact to 0.1", format="percent"), "months": st.column_config.NumberColumn("Months", format="%d")})
            top = sc.head(12).reset_index().rename(columns={"index": "model"}).iloc[::-1]
            fig = px.bar(top, x="RMSE / nowcast", y="model", orientation="h", color=np.where(top["RMSE / nowcast"] < 1, "beats the nowcast", "does not"),
                         color_discrete_map={"beats the nowcast": TEAL, "does not": GREY}, title="Error relative to the nowcast (below 1 beats it)")
            fig.add_vline(x=1, line_dash="dash", line_color=INK); show(fig, c2, height=380, yaxis_title="", xaxis_title="RMSE ÷ nowcast RMSE")
            how_to_read("**Typical error** is the root-mean-square forecast error in percentage points. **Error vs nowcast** below 1 means the model beat the Cleveland Fed's free forecast. "
                        "**R²** = 1 − SSE/SST: 1 is perfect, 0 matches always forecasting the average print, and negative is worse than that. Negative R² is common for monthly inflation, which is very noisy; "
                        "whether a model is useful is the comparison with the nowcast, not R².")
        if len(d):
            show(time_axes(px.bar(d.assign(error=d[acol] - d[fcol]), x="date", y="error", color="model", barmode="group", color_discrete_map=SERIES_COLOR,
                                  title="Forecast misses (published minus forecast, pp)")), height=320, yaxis_title="pp")
    with tab_drv:
        L = D["ladder"]
        c1, c2 = st.columns([1.15, 1])
        if has(L):
            with c1:
                lm = st.radio("Model", list(L.model.unique()), horizontal=True, key="p2_ladder_model")
                Mx = L[L.model == lm].pivot(index="set", columns="target", values="ratio").reindex(index=list(dict.fromkeys(L["set"])))
                show(px.imshow(Mx, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=0.7, zmax=1.3, aspect="auto"), height=390, title="Error vs nowcast as information is added (green beats it)",
                     coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        with c2:
            for title, key in (("Does removing FSBI hurt? (Clark-West p below 0.10 = FSBI helps)", "ablation"), ("With vs without FSBI, by benchmark", "anchor")):
                if has(D[key]):
                    t_ = D[key][D[key].target == target] if "target" in D[key] else D[key]
                    st.markdown(f"**{title}**"); st.dataframe(t_.round(3), hide_index=True, **WIDE)
        IM = D["imp"][D["imp"].target == target] if has(D["imp"]) else pd.DataFrame()
        st.markdown(f"#### Which inputs the models rely on: {target}")
        if IM.empty:
            empty_note("No feature importance in this export.")
        else:
            views = list(IM["view"].unique()); head = [v for v in ("Overall (mean of models)", "Online-selected forecast") if v in views]; views = head + [v for v in views if v not in head]
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            view_a = c1.selectbox("View", views, index=0, key="imp_view"); view_b = c2.selectbox("Compare with", ["(none)"] + views, index=min(2, len(views)), key="imp_compare")
            top_n = c3.slider("Top inputs", 5, 30, 12, key="imp_n")
            blocks = st.multiselect("Input groups", list(BLOCK_COLOR), default=list(BLOCK_COLOR), key="imp_blocks")
            cols = st.columns(2 if view_b != "(none)" else 1)
            for col, v in zip(cols, [view_a] + ([view_b] if view_b != "(none)" else [])):
                x = IM[(IM["view"] == v) & IM.block.isin(blocks) & (IM.share > 0)].sort_values("share", ascending=False).head(top_n).iloc[::-1]
                if x.empty:
                    col.info(f"{v}: no inputs selected (the model predicts the benchmark plus a constant).")
                else:
                    show(px.bar(x.assign(pct=100 * x.share), x="pct", y="feature", orientation="h", color="block", color_discrete_map=BLOCK_COLOR, category_orders={"feature": list(x.feature[::-1])}),
                         col, height=150 + 24 * len(x), title=v, xaxis_title="share of importance (%)", yaxis_title="")
            share = IM.groupby(["view", "block"]).share.sum().unstack().reindex(index=views[::-1], columns=list(BLOCK_COLOR)).fillna(0) * 100
            if share.values.sum() > 0:
                show(go.Figure([go.Bar(y=share.index, x=share[b], name=b, orientation="h", marker_color=c) for b, c in BLOCK_COLOR.items()]), height=420, barmode="stack",
                     title="Where each model's signal comes from (%)", xaxis_title="share of importance (%)")
            how_to_read("Permutation importance: how much a fitted model's error rises when one input is shuffled. It describes what each model leans on; "
                        "whether an input *improves* forecasts out of sample is the ladder and the Clark-West test above.")
        with st.expander("Granular models: CPI components and regional CPI"):
            for title, key in (("CPI components from matched FSBI sub-sectors", "comp"), ("Regional CPI from regional FSBI", "regtest")):
                if has(D[key]):
                    st.markdown(f"**{title}**"); st.dataframe(D[key].round(3), hide_index=True, **WIDE)
    with tab_fsbi:
        F = D["fsbi"]; c1, c2, c3, c4 = st.columns(4)
        lvl = c1.selectbox("FSBI level", [l for l in ["National", "Sector", "Subsector", "Region"] if (F.level == l).any()], key="p2_lvl"); names = sorted(F.loc[F.level == lvl, "name"].unique())
        nm = c2.selectbox("Series", names, key="p2_name"); met = c3.selectbox("FSBI metric", ["ticket_index", "txn_index", "sales_index"], format_func=lambda x: {"ticket_index": "Ticket", "txn_index": "Transactions", "sales_index": "Sales"}[x], key="p2_met")
        lag = c4.slider("FSBI leads by (months)", 0, 3, 0, key="p2_lag")
        fs_all = with_measure(F[(F.level == lvl) & (F.name == nm)], met, "MoM %" if freq2 == "MoM" else "YoY %").set_index("date")["value"]
        tg_all = I[(I.target == target) & (I.model == HIST)].set_index("date")[acol]
        both = pd.concat([fs_all.shift(lag).rename("fsbi"), tg_all.rename("target")], axis=1).dropna(); both = both[(both.index >= t0) & (both.index <= t1)]
        if len(both) > 5:
            fig = go.Figure(); fig.add_trace(go.Scatter(x=both.index, y=both.target, name=f"{target} {freq2} (left)", line=dict(color=INK, width=2.5)))
            fig.add_trace(go.Scatter(x=both.index, y=both.fsbi, name=f"FSBI {nm} {met.split('_')[0]}, {lag}m earlier (right)", yaxis="y2", line=dict(color=ORANGE, width=2)))
            c1, c2 = st.columns([2.2, 1])
            show(time_axes(fig), c1, height=390, yaxis2=dict(overlaying="y", side="right", title=f"FSBI {freq2} %", showgrid=False), yaxis_title=f"{target} {freq2} %",
                 title=f"Correlation {both.corr().iloc[0, 1]:+.2f} over {len(both)} months")
            lc = pd.DataFrame({"lead": range(4), "r": [pd.concat([fs_all.shift(l), tg_all], axis=1).dropna().loc[t0:t1].corr().iloc[0, 1] for l in range(4)]})
            show(px.bar(lc, x="lead", y="r", range_y=(-1, 1), color_discrete_sequence=[ORANGE]), c2, height=390, title="Correlation by lead", xaxis_title="FSBI leads by (months)", yaxis_title="correlation")
        else:
            empty_note("Too few overlapping months for this FSBI series and target in the timeline.")

# ================================================================== 3. trading strategies
elif page.startswith("3"):
    header("Trading results", "FSBI strategies by family, on one set of costs and significance tests, with a drill-down for each strategy.")
    T, TR, LBD, WRET = D["strat"], D["trades"], D["lb"], D["wret"]
    if T.empty:
        empty_note("No strategies in this export."); st.stop()
    if len(LBD):
        LBD = LBD.assign(Verdict=LBD.apply(verdict, axis=1))
        n_sig = int((LBD["Holm p"] < 0.10).sum()); n_prom = int((LBD.Verdict == "Promising").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strategies tested", len(LBD)); m2.metric("Significant after correction", n_sig, help="Holm-adjusted p below 0.10.")
        m3.metric("Promising (raw p < 0.10)", n_prom); m4.metric("Best design-period Sharpe", f"{LBD.Sharpe.max():.2f}")
        st.subheader("Family comparison")
        FA = LBD.groupby("family").agg(strategies=("strategy", "size"), median_sharpe=("Sharpe", "median"), best_sharpe=("Sharpe", "max"), promising=("Verdict", lambda v: int(v.isin(["Promising", "Significant"]).sum())),
                                       best=("strategy", lambda x: x.loc[LBD.loc[x.index, "Sharpe"].idxmax()])).reset_index()
        if "Sharpe (holdout)" in LBD:
            FA["live_sharpe"] = FA.family.map(LBD.groupby("family")["Sharpe (holdout)"].median())
        c1, c2 = st.columns([1, 1.15])
        fl = FA.melt(id_vars="family", value_vars=["median_sharpe", "best_sharpe"], var_name="stat", value_name="Sharpe").replace({"median_sharpe": "Median strategy", "best_sharpe": "Best strategy"})
        show(px.bar(fl.iloc[::-1], x="Sharpe", y="family", color="stat", orientation="h", barmode="group", color_discrete_map={"Median strategy": NAVY, "Best strategy": ORANGE}, title="Design-period Sharpe by family").add_vline(x=0, line_color=INK),
             c1, height=150 + 34 * len(FA), yaxis_title="")
        Tm = T[(T.date >= t0) & (T.date <= t1 + pd.offsets.MonthEnd(1))].groupby(["family", "date"], as_index=False).ret.mean().sort_values("date")
        if len(Tm):
            Tm["growth"] = Tm.groupby("family").ret.transform(lambda r: (1 + r).cumprod())
            st0 = Tm.groupby("family", as_index=False).date.min().assign(date=lambda x: x.date - pd.offsets.MonthBegin(1), ret=0.0, growth=1.0)
            Tm = pd.concat([st0, Tm], ignore_index=True).sort_values(["family", "date"])
            show(shade_live(time_axes(px.line(Tm, x="date", y="growth", color="family", title="Growth of $1: equal-weight average of each family's strategies")).add_hline(y=1, line_color=GREY), Tm.date.max()), c2, height=150 + 34 * len(FA), yaxis_title="log scale", yaxis_type="log",
                 legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, font=dict(size=10)))
        st.dataframe(FA, hide_index=True, **WIDE, column_config={"family": st.column_config.TextColumn("Family", width="medium"), "strategies": st.column_config.NumberColumn("Strategies"),
                     "median_sharpe": st.column_config.NumberColumn("Median Sharpe", format="%.2f"), "best_sharpe": st.column_config.NumberColumn("Best Sharpe", format="%.2f"), "promising": st.column_config.NumberColumn("Promising or better"),
                     "live_sharpe": st.column_config.NumberColumn("Median live Sharpe", format="%.2f", help=f"Since {DESIGN_END:%b %Y}."), "best": st.column_config.TextColumn("Best strategy", width="large")})
        with st.expander("What each family trades"):
            for f_, txt in meta.get("family_about", {}).items():
                st.markdown(f"**{f_}** {txt}")
        st.subheader("Leaderboard")
        c1, c3, c4 = st.columns([1.2, 1.2, 1.4])
        RANK_LABEL = {"Sharpe": "Sharpe ratio", "Sortino": "Sortino ratio", "Calmar": "Calmar ratio", "ann. return (%)": "Return per year", "max DD (%)": "Max drawdown (smallest first)",
                      "profit factor": "Profit factor", "trade win rate": "Win rate", "Jensen's alpha (ann. %)": "Jensen's alpha (CAPM)", "two-factor alpha (ann. %)": "Two-factor alpha (SPY + IEF)", "appraisal ratio": "Appraisal ratio"}
        rank_opts = [RANK_LABEL[m] for m in RANK_LABEL if m in LBD]           # friendly labels as the options themselves
        rank_by = {v: k for k, v in RANK_LABEL.items()}[c1.selectbox("Rank by", rank_opts, key="lb_rank")]
        fams_all = sorted(LBD.family.dropna().unique())
        fam_sel = st.pills("Families", fams_all, selection_mode="multi", default=fams_all, key="lb_src") if hasattr(st, "pills") else st.multiselect("Families", fams_all, default=fams_all, key="lb_src")
        sig_only = c3.checkbox("Significant only", value=False, key="lb_sig", help="Only strategies whose Holm-adjusted p-value is below 0.10.")
        topn = c4.slider("Show top", 3, max(4, min(45, len(LBD))), min(12, len(LBD)), key="lb_n")
        L = LBD[LBD.family.isin(fam_sel or fams_all)]
        if sig_only:
            L = L[L["Holm p"] < 0.10]
        L = L.dropna(subset=[rank_by]).sort_values(rank_by, ascending=False).head(topn).reset_index(drop=True)
        selection(Families=f"all {len(fams_all)}" if len(fam_sel or fams_all) == len(fams_all) else short_list([f.split(".")[0] for f in fam_sel], 12), Ranked_by=RANK_LABEL[rank_by], Filter="significant only" if sig_only else "all verdicts", Rows=f"top {len(L)}",
                  Statistics=f"design period, before {DESIGN_END:%b %Y}")
        if len(L):
            all_cols = st.toggle("Show every statistic", value=False, key="lb_all")
            base_cols = ["strategy", "Verdict", "Sharpe", "ann. return (%)", "max DD (%)", "trade win rate", "Holm p", "Sharpe (holdout)"]
            extra_cols = ["family", "trades", "Sortino", "Calmar", "profit factor", "bootstrap p", "reality-check p", "placebo p", "Sharpe, 1-day delay", "beta to SPY", "months (holdout)", "primary", "Holm p (primary)", "Jensen's alpha (ann. %)", "Jensen's alpha t", "Jensen's alpha Holm p", "CAPM beta", "appraisal ratio", "two-factor alpha (ann. %)", "two-factor alpha t", "bond beta", "market-adjusted return (ann. %)", "Jensen's alpha (holdout, ann. %)"]
            tb = L[[c for c in base_cols + (extra_cols if all_cols else []) if c in L]].copy(); tb.insert(0, "#", range(1, len(tb) + 1))
            if "trade win rate" in tb:
                tb["trade win rate"] = 100 * tb["trade win rate"]
            ccfg = {"strategy": st.column_config.TextColumn("Strategy", width="large"), "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "ann. return (%)": st.column_config.NumberColumn("Return / yr", format="%.1f%%"), "max DD (%)": st.column_config.NumberColumn("Max drawdown", format="%.1f%%"),
                    "trade win rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"), "Holm p": st.column_config.NumberColumn("Adjusted p", format="%.3f", help="Holm-adjusted for every strategy tried."),
                    "Sharpe (holdout)": st.column_config.NumberColumn("Live Sharpe", format="%.2f", help=f"Since {DESIGN_END:%b %Y}, never used to choose anything."),
                    "bootstrap p": st.column_config.NumberColumn("Raw p", format="%.3f"), "reality-check p": st.column_config.NumberColumn("Reality-check p", format="%.3f"),
                    "placebo p": st.column_config.NumberColumn("Placebo p", format="%.3f"), "Sharpe, 1-day delay": st.column_config.NumberColumn("Sharpe if 1 day late", format="%.2f"),
                    "Sortino": st.column_config.NumberColumn(format="%.2f"), "Calmar": st.column_config.NumberColumn(format="%.2f"), "profit factor": st.column_config.NumberColumn("Profit factor", format="%.2f"),
                    "beta to SPY": st.column_config.NumberColumn("Beta to SPY", format="%.2f")}
            ev = st.dataframe(tb, hide_index=True, on_select="rerun", selection_mode="single-row", key="lb_table", column_config=ccfg, height=min(36 * len(tb) + 40, 560), **WIDE)
            with st.expander("Chart view"):
                bar = L.iloc[::-1].assign(label=[f"{n[:60]}" for n in L.iloc[::-1].strategy])
                fig = px.bar(bar, x=rank_by, y="label", orientation="h", color="Verdict", color_discrete_map={"Significant": GREEN, "Promising": AMBER, "Inconclusive": GREY, "Negative": RED},
                             title=f"Top {len(L)} by {RANK_LABEL[rank_by].lower()}")
                show(fig, height=max(330, 30 * len(L) + 140), yaxis_title="", xaxis_title=RANK_LABEL[rank_by])
            rows_sel = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
            if rows_sel and st.session_state.get("lb_applied") != L.at[rows_sel[0], "strategy"]:
                chosen = L.at[rows_sel[0], "strategy"]; fam_c = T.loc[T.strategy == chosen, "family"]
                if len(fam_c):
                    st.session_state["p3_family"], st.session_state["p3_strategy"] = fam_c.iloc[0], chosen
                st.session_state["lb_applied"] = chosen
            st.caption(f"Statistics cover the design period (before {DESIGN_END:%b %Y}); Live Sharpe is the record since. Select a row for detail.")
            how_to_read("**Verdicts.** *Significant*: survives the Holm correction for the number of strategies tried. *Promising*: positive with a raw p-value below 0.10, but not after the correction. "
                        "*Inconclusive*: positive but indistinguishable from luck. *Negative*: lost money in the design period.\n\n"
                        "**Extra tests** (toggle *Show every statistic*): the *reality-check p* asks how often the best strategy on a board with no real edge would look this good; the *placebo p* shuffles the signal "
                        "to check that the signal, not the instruments, earned the return; *Sharpe if 1 day late* tests sensitivity to execution timing.\n\n"
                        "**Research basis by family:** " + "; ".join(f"{k}: {v}" for k, v in meta.get("references", {}).items()))
        else:
            empty_note("No strategy matches these filters.")
        P = D["port"]
        if has(P):
            with st.expander("Combined portfolios, correlations, risk rules and FSBI attribution"):
                pp = P[(P.date >= t0) & (P.date <= t1)].set_index("date").dropna(how="all")
                if len(pp):
                    c1, c2 = st.columns([1.4, 1])
                    show(time_axes(px.line((1 + pp.fillna(0)).cumprod(), labels={"value": "growth of $1", "variable": ""}, title="Combined portfolios (weights set in advance)")).add_hline(y=1, line_color=GREY), c1, height=340)
                    c2.dataframe(pd.DataFrame({c: metrics(pp[c]) for c in pp}).T.round(2), **WIDE)
                    c2.caption("The family-balanced portfolio is the single pre-specified test of whether FSBI, used every way here, makes money.")
                Rm = piv(T[(T.date >= t0) & (T.date <= t1)], "date", "strategy", "ret")
                if Rm.shape[1] > 2 and len(Rm) > 8:
                    C = Rm.corr(min_periods=8); ids = {}
                    for n in C.index:
                        lab_ = n.split(".")[0] if "." in n[:4] else n[:22]
                        while lab_ in ids.values():
                            lab_ += "'"
                        ids[n] = lab_
                    show(px.imshow(C.rename(index=ids, columns=ids), color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto"), height=520, title="How strategies move together (correlation of monthly returns)")
                for title, key in (("Risk rules: stop-loss and volatility targeting", "risk"), ("FSBI version vs the same strategy without FSBI", "attrib")):
                    if has(D[key]):
                        st.markdown(f"**{title}**"); st.dataframe(D[key].round(3), hide_index=True, **WIDE)
    st.divider()
    if has(D["fi_grid"]):
        with st.expander("Fixed income prototype grid (notebook 4.L)"):
            fg = D["fi_grid"]; gp = fg.pivot_table(index="pair", columns="signal type", values="Sharpe", aggfunc="first")
            c1, c2 = st.columns([1, 1.2])
            show(px.imshow(gp, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", text_auto=".2f", labels={"x": "", "y": "", "color": "Sharpe"}), c1, height=160 + 24 * len(gp), title="Annualised Sharpe by pair and signal type (gross, design period)")
            c2.dataframe(fg.sort_values("Sharpe", ascending=False).round(3), hide_index=True, **WIDE)
            ind_ = ~fg["signal type"].str.startswith("C")
            c2.caption(f"{int(ind_.sum())} independent backtests (type C is type A reversed, shown for reference); {int((fg.p[ind_] < 0.10).sum())} at raw p < 0.10, {int((fg['Holm p'] < 0.10).sum())} after the Holm adjustment. "
                       "Pairs are not duration-matched; the registered family L strategies are.")
    if has(D["reversals"]) and "reversed Sharpe" in D["reversals"]:
        with st.expander(f"Reversal analysis: {len(D['reversals'])} strategies with a strongly negative Sharpe"):
            rv = D["reversals"]
            st.dataframe(rv.drop(columns=["why"], errors="ignore"), hide_index=True, **WIDE, column_config={"strategy": st.column_config.TextColumn("Strategy", width="large"), "diagnosis": st.column_config.TextColumn("Diagnosis", width="large"),
                         **{c: st.column_config.NumberColumn(format="%.2f") for c in rv.columns if c not in ("strategy", "family", "diagnosis", "why", "holds live", "months (live)")}})
            for _, r in rv[rv.diagnosis.astype(str).str.startswith("opposite")].iterrows():
                st.markdown(f"**{r.strategy}**: reversed Sharpe {r['reversed Sharpe']:.2f}, live {r['reversed Sharpe (live)']:.2f} ({'holds' if r['holds live'] else 'does not hold'} in the live period). {r.get('why', '')}")
            st.caption("Each loser is rerun with positions reversed, costs still paid. Drift: the loss came from the average position, not from timing. Opposite sign: the instrument moved against the signal. "
                       "A reversal chosen after the fact is data-mined; it counts only if it also holds in the live period.")
    st.subheader("Strategy drill-down")
    c1, c2 = st.columns([1, 2.2])
    fams = sorted(T.family.unique())
    if st.session_state.get("p3_family") not in fams:
        st.session_state.pop("p3_family", None)
    fam = c1.selectbox("Family", fams, key="p3_family", **({} if "p3_family" in st.session_state else {"index": fams.index("C. Sector rotation") if "C. Sector rotation" in fams else 0}))
    pool = T[T.family == fam]; rk = pool.groupby("strategy").ret.apply(lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else -9).sort_values(ascending=False)
    if st.session_state.get("p3_strategy") not in list(rk.index):
        st.session_state.pop("p3_strategy", None)
    strat = c2.selectbox("Strategy (best first)", list(rk.index), format_func=lambda n: f"{n}   (Sharpe {rk[n]:.2f})", key="p3_strategy")
    S0 = pool[pool.strategy == strat].set_index("date").sort_index(); r_all = S0.ret; r_win = r_all[(r_all.index >= t0) & (r_all.index <= t1 + pd.offsets.MonthEnd(1))]
    spec = SPECS.get(strat, {"type": "fixed"}); k_ = kid(strat)
    instr = (", ".join(f"{'long' if w > 0 else 'short'} {asset_name(a)}" for a, w in spec.get("legs", {}).items()) if spec.get("legs")
             else ("rank the assets, long the top group and short the bottom group" if spec["type"] == "xs" else "see the notebook"))
    m = metrics(r_win)
    with st.container(border=True):
        st.markdown(f"#### {strat}")
        st.markdown(meta.get("family_about", {}).get(fam, ""))
        st.markdown(f"**Trade:** {instr} when the signal is positive (reversed when negative) · **Held:** {spec.get('window', 'see the notebook')} · **Months shown:** {len(r_win)} of {len(r_all)}")
        mc = st.columns(6)
        for c, (k, v, hlp) in zip(mc, [("Sharpe", fnum(m.get("Sharpe")), GLOSSARY["Sharpe ratio"]), ("Return / yr", fnum(m.get("ann. return %"), "{:+.1f}%"), None),
                                       ("Win rate", fnum(m.get("win rate"), "{:.0%}"), None), ("Total return", fnum(m.get("ROI %"), "{:+.1f}%"), None),
                                       ("Volatility / yr", fnum(m.get("ann. vol %"), "{:.1f}%"), None), ("Max drawdown", fnum(m.get("max drawdown %"), "{:.1f}%"), GLOSSARY["Drawdown"])]):
            c.metric(k, v, help=hlp)
        if len(LBD) and (LBD.strategy == strat).any():
            lr = LBD[LBD.strategy == strat].iloc[0]; rank_ = int((LBD.Sharpe > lr.Sharpe).sum()) + 1
            plain = {"Significant": "its result is unlikely to be luck even after allowing for every strategy tried.",
                     "Promising": "its result looks real on its own, but not after allowing for every strategy tried.",
                     "Inconclusive": "it made money, but no more than luck would explain.", "Negative": "it lost money in the design period."}[lr.Verdict]
            live = f" Since {DESIGN_END:%b %Y} (live) its Sharpe is {lr['Sharpe (holdout)']:.2f}." if "Sharpe (holdout)" in lr and pd.notna(lr["Sharpe (holdout)"]) else ""
            st.info(f"**{lr.Verdict}** · rank {rank_} of {len(LBD)} by Sharpe ({lr.Sharpe:.2f}, {int(lr.trades)} trades, adjusted p = {lr['Holm p']:.3f}): {plain}{live}")
    selection(Family=fam, Strategy=strat.split(". ", 1)[0] if ". " in strat[:5] else strat[:30], Charts=f"{t0:%b %Y} to {t1:%b %Y}", Position=f"${POSITION:,.0f}")
    tab_bt, tab_lab, tab_proj = st.tabs(["Backtest", "Signal thresholds", "Monte Carlo projection"])
    lab = {"ret": None, "pos": None, "now_pos": None, "now_text": None, "bench": None, "pos_label": None}
    base = TR[TR.strategy == strat].set_index("date").ret.sort_index() if has(TR) else pd.Series(dtype=float)
    with tab_bt:
        compare = st.multiselect("Compare with other strategies", [s for s in T.strategy.unique() if s != strat], key=f"cmp_{k_}")
        cmp = T[T.strategy.isin([strat] + compare) & (T.date >= t0) & (T.date <= t1 + pd.offsets.MonthEnd(1))].sort_values("date")
        if len(cmp):
            cmp = cmp.assign(growth=cmp.groupby("strategy").ret.transform(lambda r: (1 + r).cumprod()))
            start_ = cmp.groupby("strategy", as_index=False).date.min().assign(date=lambda x: x.date - pd.offsets.MonthBegin(1), ret=0.0, growth=1.0)
            cmp = pd.concat([start_, cmp], ignore_index=True).sort_values(["strategy", "date"]); c1, c2 = st.columns(2)   # every curve starts at $1
            show(shade_live(time_axes(px.line(cmp, x="date", y="growth", color="strategy", title="Growth of $1, net of costs")).add_hline(y=1, line_color=GREY), cmp.date.max()), c1, height=380, yaxis_title="")
            dd = cmp.assign(drawdown=cmp.groupby("strategy").growth.transform(lambda c: 100 * (c / c.cummax() - 1)))
            show(time_axes(px.area(dd, x="date", y="drawdown", color="strategy", title="Drawdown from the previous peak (%)")).update_traces(stackgroup=None, fill="tozeroy"), c2, height=380, yaxis_title="%")
            if len(r_win):
                show(time_axes(px.bar(x=r_win.index, y=r_win.values * 100, color=np.where(r_win.values > 0, "gain", "loss"), color_discrete_map={"gain": TEAL, "loss": RED}, labels={"y": "%"},
                                      title="Result each month (%)")), height=300, showlegend=False)
        else:
            empty_note()
        DDy = D["daily"]
        if has(DDy) and (DDy.strategy == strat).any():
            dd_ = DDy[(DDy.strategy == strat) & (DDy.date >= t0) & (DDy.date <= t1 + pd.offsets.MonthEnd(1))].sort_values("date")
            if len(dd_) > 3:
                show(time_axes(px.line(dd_.assign(pnl=((1 + dd_.ret).cumprod() - 1) * POSITION), x="date", y="pnl", title=f"Day by day: cumulative P&L on ${POSITION:,.0f}",
                                       color_discrete_sequence=[ORANGE])).add_hline(y=0, line_color=GREY), height=300, yaxis_title="$")
    with tab_lab:
        if spec["type"] == "timing" and has(D["ssig"]) and has(WRET):
            s_all = D["ssig"][D["ssig"].strategy == strat].set_index("date").value.sort_index(); legs = spec["legs"]; W = piv(WRET[WRET.window == spec["window"]], "date", "ticker", "ret")
            if set(legs) <= set(W.columns) and len(s_all):
                c1, c2, c3, c4 = st.columns(4)
                hi = float(max(0.02, round(s_all.abs().quantile(0.9), 2)))
                thr = c1.slider("Only trade when the signal is stronger than", 0.0, hi, float(min(hi, spec.get("deadband", 0.0))), 0.01, key=f"thr_{k_}")
                sizing = c2.radio("Position size", ["Sign (+1 / -1)", "Proportional (capped at 2 sd)"], key=f"size_{k_}")
                cost = c3.slider("Cost per side (bp)", 0.0, 20.0, float(1e4 * meta.get("cost_per_side", 0.0005)), 0.5, key=f"cost_{k_}") / 1e4
                flip = c4.checkbox("Reverse the trade", value=False, key=f"flip_{k_}")
                unit = (W[list(legs)] * pd.Series(legs)).sum(axis=1, min_count=len(legs)); gross = sum(abs(v) for v in legs.values())
                d_ = pd.concat([s_all.rename("s"), unit.rename("u")], axis=1).dropna()
                size = np.sign(d_.s) if sizing.startswith("Sign") else (d_.s / s_all.std()).clip(-2, 2) / 2
                pos = size.where(d_.s.abs() > thr, 0.0) * (-1 if flip else 1)
                lab.update(ret=pos * d_.u - 2 * cost * gross * pos.abs(), pos=np.sign(pos), bench=(d_.u - 2 * cost * gross).rename("always hold the trade"),
                           pos_label={1.0: "hold the trade" if not flip else "reversed (signal positive)", 0.0: "flat", -1.0: "reversed" if not flip else "hold the trade (signal negative)"})
                s_now = s_all.dropna()
                if len(s_now):
                    now = float(s_now.iloc[-1]); lab["now_pos"] = float(np.sign(now) * (abs(now) > thr) * (-1 if flip else 1))
                    lab["now_text"] = f"Latest signal ({s_now.index[-1]:%b %Y}): **{now:+.2f}** (trades when its size exceeds {thr:.2f})"
        elif spec["type"] == "xs" and has(D["xs"]) and has(WRET):
            S = piv(D["xs"][D["xs"].signal == strat], "date", "asset", "value"); R = piv(WRET[WRET.window == spec["window"]], "date", "ticker", "ret"); S = S[[c for c in S if c in R]]
            if S.shape[1] >= 4:
                c1, c2, c3 = st.columns(3)
                mode = c1.selectbox("Construction", ["long-short", "long-only vs equal-weight universe"], key=f"mode_{k_}")
                frac = c2.slider("Share of assets in each leg", 0.1, 0.5, 1 / 3, 0.01, key=f"frac_{k_}")
                cost = c3.slider("Cost per side (bp)", 0.0, 20.0, float(1e4 * meta.get("cost_per_side", 0.0005)), 0.5, key=f"cost_{k_}") / 1e4
                out = {}
                for t in S.index.intersection(R.index):
                    row = S.loc[t][R.loc[t].reindex(S.columns).notna()]; w = xs_weights(row, frac, spec.get("min_n", 4), mode != "long-short")
                    if w:
                        out[t] = sum(R.at[t, a] * x for a, x in w.items()) - 2 * cost * (1 if mode == "long-short" else 2)
                lab["ret"] = pd.Series(out, dtype=float)
                last = S.dropna(how="all").iloc[-1].dropna()
                if len(last) >= spec.get("min_n", 4):
                    w = xs_weights(last, frac, spec.get("min_n", 4))
                    lab["now_text"] = (f"Positions from the latest FSBI month ({S.dropna(how='all').index[-1]:%b %Y}): long **{', '.join(asset_name(a) for a, x in w.items() if x > 0)}**"
                                       + ("" if mode != "long-short" else f"; short **{', '.join(asset_name(a) for a, x in w.items() if x < 0)}**"))
        if lab["ret"] is None:
            st.caption("This strategy is priced inside the notebook (Kalshi contracts, modelled options, multi-leg books), so these controls adjust its results directly.")
            c1, c2, c3, c4, c5 = st.columns(5)
            flip = c1.checkbox("Reverse the trade", value=False, key=f"gflip_{k_}"); scale = c2.slider("Position size (x)", 0.25, 3.0, 1.0, 0.25, key=f"gscale_{k_}")
            extra = c3.slider("Extra cost per trade (bp)", 0.0, 50.0, 0.0, 1.0, key=f"gcost_{k_}"); vt = c4.slider("Volatility target (% / yr, 0 = off)", 0, 30, 0, 1, key=f"gvt_{k_}")
            cap = c5.slider("Loss cap per trade (%, 0 = off)", 0, 20, 0, 1, key=f"gcap_{k_}")
            src = base if len(base) else r_all
            r = src * (-1 if flip else 1) * scale - (extra / 1e4) * (src != 0)
            if vt:
                trail = src.rolling(12, min_periods=6).std().shift(1) * np.sqrt(12); r = r * (vt / 100 / trail).clip(upper=3).fillna(1.0)
            if cap:
                r = r.clip(lower=-cap / 100); st.caption("The loss cap assumes a stop could have been filled at that level (an approximation).")
            lab["ret"] = r
        ret = lab["ret"].dropna(); ret = ret[(ret.index >= t0) & (ret.index <= t1)]
        if len(ret) < 3:
            st.warning("Too few months in the timeline for this setting.")
        else:
            curve = pd.DataFrame({"your version": (1 + ret).cumprod()})
            if len(base):
                curve["as backtested"] = (1 + base.reindex(ret.index).fillna(0)).cumprod()
            if lab["bench"] is not None:
                curve[lab["bench"].name] = (1 + lab["bench"].reindex(ret.index).fillna(0)).cumprod()
            c1, c2 = st.columns([1.6, 1])
            show(time_axes(px.line(curve, title="Your version vs the original", labels={"value": "growth of $1", "variable": ""},
                                   color_discrete_sequence=[ORANGE, INK, GREY])).add_hline(y=1, line_color=GREY), c1, height=380)
            mt = pd.DataFrame({"your version": metrics(ret), **({"as backtested": metrics(base.reindex(ret.index).fillna(0))} if len(base) else {}),
                               **({lab["bench"].name: metrics(lab["bench"].reindex(ret.index))} if lab["bench"] is not None else {})}).T
            c2.dataframe(mt.round(2), **WIDE)
            c2.metric(f"Your version on ${POSITION:,.0f} per trade", f"${POSITION * ((1 + ret).prod() - 1):+,.0f}", f"{int((ret != 0).sum())} trades", delta_color="off")
            if lab["pos"] is not None and lab["pos_label"]:
                pdf = pd.DataFrame({"date": lab["pos"].index, "position": lab["pos"].map(lab["pos_label"])}).dropna(); pdf = pdf[(pdf.date >= t0) & (pdf.date <= t1)]
                if len(pdf):
                    show(px.scatter(pdf, x="date", y="position", color="position", title="Position each month", color_discrete_sequence=[ORANGE, GREY, NAVY]), height=240, showlegend=False, xaxis_title="", yaxis_title="")
    with tab_proj:
        if lab["now_text"]:
            st.markdown(lab["now_text"])
        if lab["now_pos"] is not None and lab["pos_label"]:
            st.markdown(f"Position implied **now** with your settings: **{lab['pos_label'].get(lab['now_pos'], lab['now_pos'])}**")
        ret = lab["ret"].dropna(); ret = ret[(ret.index >= t0) & (ret.index <= t1)]
        if len(ret) < 6:
            st.info("The projection needs at least 6 months of results in the timeline.")
        else:
            c1, c2, c3 = st.columns(3); horizon = c1.slider("Months ahead", 3, 24, 12, key="mc_h"); sims = c2.slider("Simulations", 500, 5000, 2000, 500, key="mc_n")
            can_cond = lab["pos"] is not None and lab["now_pos"] is not None
            cond = c3.checkbox("Use only months with today's position", value=can_cond, disabled=not can_cond, key=f"mc_cond_{k_}")
            sample = ret
            if cond and can_cond:
                same = ret[lab["pos"].reindex(ret.index) == lab["now_pos"]]
                if len(same) >= 6:
                    sample = same; st.caption(f"Resampling {len(same)} past months that had today's position.")
                else:
                    st.caption(f"Only {len(same)} months had today's position; using all months.")
            rng = np.random.default_rng(7); vals = sample.values; n = len(vals); block = 3; paths = np.empty((sims, horizon))
            for i in range(sims):
                idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, horizon // block + 1)])[:horizon]
                paths[i] = np.cumprod(1 + vals[idx]) - 1
            q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0) * 100; x = np.arange(1, horizon + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[4], q[0][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.12)", line=dict(width=0), name="5-95% of outcomes"))
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[3], q[1][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.30)", line=dict(width=0), name="25-75%"))
            fig.add_trace(go.Scatter(x=x, y=q[2], line=dict(color=NAVY, width=2.5), name="median")); fig.add_hline(y=0, line_color=GREY)
            end = paths[:, -1] * 100; c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median outcome", f"{np.median(end):+.1f}%"); c2.metric("Chance of a loss", f"{(end < 0).mean():.0%}")
            c3.metric("Bad case (5%)", f"{np.percentile(end, 5):+.1f}%", f"${POSITION * np.percentile(end, 5) / 100:+,.0f} on ${POSITION:,.0f}", delta_color="off"); c4.metric("Good case (95%)", f"{np.percentile(end, 95):+.1f}%")
            show(fig, height=400, title=f"Range of outcomes over the next {horizon} months ({sims:,} simulations)", xaxis_title="months ahead", yaxis_title="cumulative return (%)")
            st.caption("Resamples this strategy's own monthly results in 3-month blocks. It assumes the future looks like the past: it shows the spread of outcomes, not proof of an edge.")

# ================================================================== 4. economic signals
elif page.startswith("4"):
    header("FSBI correlations", "FSBI against macro series, official sector and state data, CPI components and market assets, at leads and lags.")
    SIG, WRET = D["sig"], D["wret"]; COST = float(meta.get("cost_per_side", 0.0005))
    def ratio_chart(tab, title, where=st):
        t_ = tab.dropna(subset=["ratio"]).sort_values("ratio", ascending=False)
        if len(t_):
            col = np.where((t_.ratio < 1) & (t_["CW p"] < 0.10), "more accurate (significant)", np.where(t_.ratio < 1, "more accurate", "no improvement"))
            show(px.bar(t_, x="ratio", y="target", orientation="h", color=col, color_discrete_map={"more accurate (significant)": GREEN, "more accurate": "#9fd8c5", "no improvement": GREY},
                        title=title, labels={"ratio": "error with FSBI ÷ error without", "target": ""}).add_vline(x=1, line_dash="dash", line_color=INK), where, height=140 + 30 * len(t_))
    N, S, P, MP, LD, LP, IC, MR, EVT, MI = D["econ_nat"], D["econ_sec"], D["econ_state"], D["econ_map"], D["leads"], D["lead_prof"], D["sc_ic"], D["macro_reads"], D["event"], D["macro_ind"]
    selection(Period=f"{t0:%b %Y} to {t1:%b %Y}", Latest_FSBI_month=f"{LIVE_M:%b %Y}", Tests=f"design period, before {DESIGN_END:%b %Y}")
    tabs = st.tabs(["Macro series: leads and lags", "Market assets", "Retail sales by store type", "States", "Small-cap sectors", "CPI components and regions", "Composite indicators", "Nowcast gains", "Release-day reaction"])
    with tabs[0]:
        if has(LD):
            measures = list(dict.fromkeys(LD["FSBI measure"]))
            c1, c2 = st.columns([1, 2])
            msel = c1.selectbox("FSBI measure", measures, key="lead_measure")
            ld = LD[LD["FSBI measure"] == msel].sort_values("Granger p: FSBI -> series")
            leads_n = int((ld.verdict == "FSBI leads").sum()); two = int((ld.verdict == "two-way").sum())
            names_ = ", ".join(ld[ld.verdict == "FSBI leads"].series.head(4))
            c2.markdown(f"**{msel}** moves ahead of **{leads_n}** of {len(ld)} series" + (f" ({names_})" if names_ else "") + f", and in both directions with {two} more (Granger tests, Holm-adjusted).")
            if has(LP):
                prof = LP[LP.measure == msel].pivot_table(index="series", columns="lead", values="corr").reindex(ld.series)
                show(px.imshow(prof, color_continuous_scale="RdBu_r", zmin=-0.6, zmax=0.6, aspect="auto", text_auto=".1f", labels={"x": "months FSBI is ahead (negative = behind)", "y": "", "color": "corr"}),
                     height=170 + 26 * len(prof), title=f"Correlation of {msel} with each series k months later")
            show_tab = ld[["series", "verdict", "peak lead (months)", "corr at peak", "corr, FSBI 1 month ahead", "Granger Holm p: FSBI -> series", "Granger Holm p: series -> FSBI"]]
            st.dataframe(show_tab, hide_index=True, **WIDE, column_config={"series": st.column_config.TextColumn("Series", width="medium"), "verdict": st.column_config.TextColumn("Verdict", width="small"),
                "peak lead (months)": st.column_config.NumberColumn("Strongest at lead (months)", format="%d"), "corr at peak": st.column_config.NumberColumn("Correlation there", format="%.2f"),
                "corr, FSBI 1 month ahead": st.column_config.NumberColumn("Correlation, 1 month ahead", format="%.2f"),
                "Granger Holm p: FSBI -> series": st.column_config.NumberColumn("p: FSBI → series", format="%.3f"), "Granger Holm p: series -> FSBI": st.column_config.NumberColumn("p: series → FSBI", format="%.3f")})
            how_to_read("Each row asks whether this FSBI measure moves **before** an official series. The heat map shows the correlation of FSBI in month *t* with the series in month *t + k*: colour to the right of 0 means FSBI moves first. "
                        "The **Granger test** asks whether past FSBI helps predict the series beyond the series' own past; *FSBI leads* means that test passes (after the Holm correction) and the reverse does not. "
                        "FSBI for a month is published around the 6th of the next month, before most of these series, so even a same-month link is a timing advantage. These are in-sample associations; the "
                        "out-of-sample tests are on the *Nowcasting* tabs.")
        else:
            empty_note("No leading-indicator scan in this export (notebook section 3.9).")
    with tabs[7]:
        if has(N):
            c1, c2 = st.columns([1.2, 1]); ratio_chart(N, "Does adding FSBI make the forecast more accurate?", c1)
            c2.dataframe(N.round(3), hide_index=True, **WIDE)
            how_to_read("Each row compares a public forecast (the series' own past) with the same forecast plus FSBI, month by month out of sample. A bar left of the dashed line means FSBI made it more accurate; "
                        "dark green means the gain passes the Clark-West test. *Nowcast* rows forecast a month whose official number comes out after FSBI; *lead* rows forecast the following month.")
        else:
            empty_note("No national activity tests in this export.")
    with tabs[2]:
        if has(S):
            c1, c2 = st.columns([1.2, 1]); ratio_chart(S, "FSBI sub-sector sales as a nowcast of the Census series for the same store type", c1)
            c2.dataframe(S.round(3), hide_index=True, **WIDE)
        else:
            empty_note("No store-type tests in this export.")
    with tabs[3]:
        if has(P):
            c1, c2 = st.columns([1, 1.2])
            c1.dataframe(P.round(3), hide_index=True, **WIDE)
            c1.caption("'Relative to the national average' asks whether a state's FSBI explains how that state differs from the nation: local information no national series carries.")
            if has(MP) and MP["corr"].notna().any():
                show(px.choropleth(MP.dropna(), locations="state", locationmode="USA-states", color="corr", scope="usa", color_continuous_scale="RdBu", range_color=(-0.6, 0.6),
                                   title="How closely each state's FSBI tracks its economy (correlation)"), c2, height=400)
        else:
            empty_note("No state tests in this export.")
    with tabs[4]:
        if has(IC):
            ic = IC.copy(); ic["label"] = ic["returns"].str.replace(" sector funds", "").str.replace("small-minus-large spreads", "small minus large")
            fig = px.bar(ic, x="horizon", y="mean IC", color="label", barmode="group", text=ic["t-stat (Newey-West)"].map(lambda v: f"t={v:.1f}"),
                         color_discrete_map={"small-cap": ORANGE, "large-cap": NAVY, "small minus large": TEAL}, title="Does FSBI's sector signal predict the next months' sector returns?")
            fig.add_hline(y=0, line_color=INK, line_width=1)
            c1, c2 = st.columns([1.4, 1])
            show(fig, c1, height=380, yaxis_title="average rank correlation (IC)", xaxis_title="")
            c2.dataframe(IC.round(3), hide_index=True, **WIDE)
            how_to_read("FSBI classifies small businesses by industry. Each month the seven FSBI sector signals are compared with the next months' returns of the matching **small-cap** sector funds, the **large-cap** "
                        "funds for the same sectors, and the **small-minus-large** spread. A positive IC means sectors where small businesses did better went on to outperform. "
                        "If FSBI's information reaches small firms' prices later, the small-cap and spread bars should be higher than the large-cap ones. |t| above 2 is the usual bar for a reliable effect. "
                        "The strategies built on this are family K on the Trading strategies page.")
        else:
            empty_note("No small-cap tests in this export (notebook section 4.K).")
    with tabs[5]:
        for title, key in (("CPI components from matched FSBI sub-sector tickets", "comp"), ("Regional CPI from regional FSBI", "regtest")):
            if has(D[key]):
                st.markdown(f"**{title}**"); t_ = D[key].rename(columns={D[key].columns[0]: "target"})
                c1, c2 = st.columns([1.2, 1]); ratio_chart(t_, "Error with FSBI ÷ error without", c1); c2.dataframe(t_.round(3), hide_index=True, **WIDE)
    with tabs[6]:
        if has(MR):
            if has(MI):
                mi = MI.copy(); mi["date"] = pd.to_datetime(mi["date"]); mi = mi[(mi.date >= t0) & (mi.date <= t1)]
                isel = st.selectbox("Indicator", [c for c in mi.columns if c != "date"], key="macro_ind")
                show(time_axes(px.line(mi, x="date", y=isel, title=isel, color_discrete_sequence=[ORANGE]).add_hline(y=float(mi[isel].mean()), line_dash="dot", line_color=GREY)), height=300, yaxis_title="")
            mr = MR.assign(col=MR.target + " | " + MR.horizon); tt = mr.pivot_table(index="FSBI indicator", columns="col", values="t-stat", aggfunc="first")
            tt.index = [i.split(" (")[0] for i in tt.index]
            show(px.imshow(tt, color_continuous_scale="RdBu_r", zmin=-4, zmax=4, aspect="auto", text_auto=".1f", labels={"x": "", "y": "", "color": "t"}), height=260 + 40 * len(tt),
                 title="t-statistic of each indicator as a predictor of the target (Newey-West)", xaxis_tickangle=-35)
            st.dataframe(MR.sort_values("p").round(3), hide_index=True, **WIDE)
            how_to_read("Four indicators built from the sector-by-state panel (spending mix, cross-state dispersion, ticket minus transaction growth, sales acceleration), each measured against its own previous six months and tested as a predictor of the next one and three months of "
                        "six macro series (controlling for the target's current value) and of three market spreads over the following FSBI-to-FSBI window. Holm p adjusts for every test in the table; shift p is the share of circular shifts of the indicator that do as well, a check that does not rely on the standard errors. Design period only.")
        else:
            empty_note("No composite-indicator tests in this export (notebook section 3.10).")
    with tabs[8]:
        if has(EVT):
            c1, c2 = st.columns([1, 1.3])
            ev_ = EVT.sort_values("ratio")
            show(px.bar(ev_, x="ratio", y="asset", orientation="h", title="Release-day volatility relative to other days", color_discrete_sequence=[ORANGE], labels={"ratio": "mean absolute move, release day / other days", "asset": ""}).add_vline(x=1, line_dash="dash", line_color=INK), c1, height=150 + 34 * len(ev_))
            c2.dataframe(EVT.round(2), hide_index=True, **WIDE)
            st.caption("Same-day price move regressed on the FSBI transaction-growth surprise. No reaction and a ratio near 1 indicate the release is not yet treated as market news.")
        else:
            empty_note("No release-day study in this export (notebook section 3.10).")
    with tabs[1]:
        if SIG.empty or WRET.empty:
            empty_note("No signals or returns in this export.")
        else:
            sig_names = [c for c in SIG.columns if c != "date"]; tickers = sorted(WRET.ticker.unique())
            c1, c2, c3, c4 = st.columns([2.2, 1.6, 1, 1])
            s_name = c1.selectbox("Signal", sig_names, key="lab_signal"); window = c2.selectbox("Holding period", list(WRET.window.unique()), key="lab_window")
            long = c3.selectbox("Buy", tickers, index=tickers.index("XLY") if "XLY" in tickers else 0, key="lab_long", format_func=asset_name)
            short = c4.selectbox("Against (sell)", ["(none)"] + tickers, index=(tickers.index("XLP") + 1) if "XLP" in tickers else 0, key="lab_short", format_func=lambda a: a if a == "(none)" else asset_name(a))
            lag_m = st.slider("Lag: months between the FSBI signal and the trade", 0, 6, 0, key="lab_lag", help="0 = trade the window that opens when the signal is published; k = wait k further months.")
            s_all = SIG.set_index("date")[s_name].dropna().shift(lag_m).dropna(); W = piv(WRET[WRET.window == window], "date", "ticker", "ret")
            c1, c2, c3 = st.columns(3)
            thr = c1.slider("Only trade when the signal is stronger than", 0.0, float(max(0.02, round(s_all.abs().quantile(0.9), 2))), 0.0, 0.01, key="lab_thr")
            direction = c2.radio("Direction", ["Follow the signal", "Fade the signal"], horizontal=True, key="lab_dir"); sizing = c3.radio("Position size", ["Sign (+1 / -1)", "Proportional (capped at 2 sd)"], horizontal=True, key="lab_size")
            u = ((W[long] - W[short]) / 2 if short != "(none)" and short in W else W[long]) if long in W else pd.Series(dtype=float)
            d = pd.concat([s_all.rename("signal"), u.rename("unit")], axis=1).dropna(); d = d[(d.index >= t0) & (d.index <= t1)]
            selection(Signal=s_name, Asset=asset_name(long) + (f" vs {asset_name(short)}" if short != "(none)" else ""), Holding=window, Lag=f"{lag_m} month{'s' if lag_m != 1 else ''}", Months=len(d))
            lc = pd.Series({k: SIG.set_index("date")[s_name].shift(k).corr(u) for k in range(0, 7)}).dropna()
            if len(lc):
                show(px.bar(x=lc.index, y=lc.values, color_discrete_sequence=[NAVY], title="Correlation of the signal with the return, by lag"), height=240, xaxis_title="lag (months)", yaxis_title="correlation", xaxis_dtick=1)
            if len(d) < 6:
                empty_note("Fewer than six months with both the signal and the return.")
            else:
                pos = (np.sign(d.signal) if sizing.startswith("Sign") else (d.signal / d.signal.std()).clip(-2, 2) / 2).where(d.signal.abs() > thr, 0.0) * (1 if direction.startswith("Follow") else -1)
                r = pos * d.unit - 2 * COST * pos.abs(); tr = r[pos != 0]; mm = metrics(r)
                cs = st.columns(6)
                for c, (k, v) in zip(cs, [("Trades", f"{int((pos != 0).sum())}"), ("Average per trade", f"{1e4 * tr.mean():+.1f} bp" if len(tr) else "–"), ("Win rate", f"{(tr > 0).mean():.0%}" if len(tr) else "–"),
                                          ("t-statistic", f"{tr.mean() / tr.std() * np.sqrt(len(tr)):.2f}" if len(tr) > 2 and tr.std() > 0 else "–"), ("Sharpe", fnum(mm.get("Sharpe"))), ("Signal-return correlation", f"{d.signal.corr(d.unit):+.2f}")]):
                    c.metric(k, v)
                c1, c2 = st.columns(2)
                show(time_axes(px.line(((1 + r).cumprod() - 1) * POSITION, labels={"value": "$"}, title=f"Cumulative P&L on ${POSITION:,.0f}", color_discrete_sequence=[ORANGE])).update_traces(showlegend=False), c1, height=360)
                fig = px.scatter(d.assign(ret=100 * d.unit), x="signal", y="ret", color_discrete_sequence=[NAVY]); b = np.polyfit(d.signal, 100 * d.unit, 1); xs_ = np.linspace(d.signal.min(), d.signal.max(), 20)
                fig.add_scatter(x=xs_, y=b[0] * xs_ + b[1], mode="lines", line=dict(color=ORANGE, dash="dash"))
                show(fig, c2, height=360, title="Signal vs the return that followed", xaxis_title=s_name, yaxis_title="return over the holding period (%)", showlegend=False)
                q = pd.qcut(d.signal, 3, labels=["weak signal", "middle", "strong signal"], duplicates="drop")
                st.dataframe((100 * d.unit).groupby(q, observed=True).agg(["count", "mean", "median", "std"]).round(2).rename(columns={"count": "months", "mean": "average return (%)", "median": "median (%)", "std": "spread (sd, %)"}), **WIDE)
