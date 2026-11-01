"""FSBI x Inflation dashboard. Reads the files written by the notebook's export cell (data/dashboard/).
Run:  streamlit run dashboard/app.py      (or set FSBI_DASH_DATA to the folder with fsbi.csv, inflation.csv, ...)"""
import json, os, re
from math import erf, sqrt
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FSBI x Inflation", page_icon="📈", layout="wide")
DEFAULT = Path(os.environ.get("FSBI_DASH_DATA", Path(__file__).resolve().parent.parent / "data" / "dashboard"))
Z = {80: 1.2816, 90: 1.6449, 95: 1.9600}
ncdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
W = "stretch"

# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load(folder):
    p = Path(folder); rd = lambda n, **kw: pd.read_csv(p / n, **kw) if (p / n).exists() else pd.DataFrame()
    d = {"fsbi": rd("fsbi.csv", parse_dates=["date"]), "inf": rd("inflation.csv", parse_dates=["date"]), "strat": rd("strategies.csv", parse_dates=["date"]),
         "sig": rd("signals.csv", parse_dates=["date"]).set_index("date"), "meta": json.loads((p / "meta.json").read_text()),
         "secsig": rd("sector_signals.csv", parse_dates=["date"]), "secret": rd("sector_returns.csv", parse_dates=["date"]), "secmap": rd("sector_map.csv"),
         "single": rd("single_name.csv", parse_dates=["date"])}
    d["fsbi"]["region"] = d["fsbi"]["region"].fillna("")          # national and sector rows have no region; NaN keys would vanish from groupby
    for c in ("forecast_yoy", "actual_yoy"):
        if c not in d["inf"]: d["inf"][c] = np.nan
    return d

def with_measure(df, col, measure):
    df = df.sort_values("date").copy()
    g = df.groupby(["level", "name", "region"], dropna=False)[col]
    df["value"] = g.pct_change(fill_method=None) * 100 if measure == "MoM %" else g.pct_change(12, fill_method=None) * 100 if measure == "YoY %" else df[col]
    return df

def metrics(r, dollar=False):
    r = pd.Series(r).dropna()
    if len(r) < 2: return {"months": len(r)}
    down = r[r < 0]
    out = {"months": len(r), "Sharpe": r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan, "Sortino": r.mean() / down.std() * np.sqrt(12) if len(down) > 1 and down.std() > 0 else np.nan, "win rate": (r > 0).mean()}
    if dollar: return {**out, "total $ / contract": r.sum(), "mean $ / month": r.mean()}
    c = (1 + r).cumprod()
    return {**out, "ROI %": 100 * (c.iloc[-1] - 1), "ann. return %": 100 * (c.iloc[-1] ** (12 / len(r)) - 1), "ann. vol %": 100 * r.std() * np.sqrt(12), "max drawdown %": 100 * (c / c.cummax() - 1).min()}

def kid(name): return re.sub(r"\W+", "_", name)[:48]
def empty_note(msg="No data for this selection. Widen the timeline or change the filters."): st.info(msg)

# ------------------------------------------------------------------ sidebar (global)
st.sidebar.title("FSBI x Inflation")
folder = st.sidebar.text_input("Data folder", str(DEFAULT), help="The folder the notebook's export cell writes (data/dashboard).")
if not (Path(folder) / "fsbi.csv").exists():
    st.error(f"No dashboard data in {folder}. Run the notebook through its 'Export for the dashboard' cell, or point the box above at data/dashboard."); st.stop()
D = load(folder); meta = D["meta"]
page = st.sidebar.radio("Page", ["1. FSBI explorer", "2. Inflation forecasts", "3. Trading strategies"], key="page")
months = sorted(D["fsbi"].date.dt.strftime("%Y-%m").unique())
start_m, end_m = st.sidebar.select_slider("Timeline", options=months, value=(months[max(0, len(months) - 61)], months[-1]), key="timeline")
t0, t1 = pd.Timestamp(start_m), pd.Timestamp(end_m)
st.sidebar.caption(f"Data updated {meta['updated']} | FSBI through {meta['fsbi_last_month']} | holdout from {meta['holdout_start']}")
CUR = pd.DataFrame(meta.get("current", []))

# ================================================================== page 1
if page.startswith("1"):
    st.title("Fiserv Small Business Index: spending trends")
    F = D["fsbi"]; METRIC = {"Sales": "sales_index", "Transactions": "txn_index", "Ticket (sales per transaction)": "ticket_index"}
    cols = st.columns(3)
    for c, (lbl, col) in zip(cols, METRIC.items()):
        s = with_measure(F[F.level == "National"], col, "YoY %").dropna(subset=["value"])
        if len(s) >= 2: c.metric(f"National {lbl.split(' ')[0].lower()}, YoY", f"{s.value.iloc[-1]:+.1f}%", f"{s.value.iloc[-1] - s.value.iloc[-2]:+.1f} pp vs prior month", help=f"Latest month: {s.date.iloc[-1]:%b %Y}")
    regions = list(meta["regions"]); sectors = sorted(F.loc[F.level == "Sector", "name"].unique())
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1.1, 1.4, 1.2])
        level = c1.selectbox("Level", ["National", "Sector", "Subsector", "Region", "State", "Region x sector"], index=1, key="p1_level")
        metric = c2.radio("Metric", list(METRIC), key="p1_metric"); measure = c3.radio("Measure", ["Index", "MoM %", "YoY %"], index=2, horizontal=True, key="p1_measure")
        sel_regions = c4.multiselect("Regions", regions, default=regions, key="p1_regions", disabled=level not in ("Region", "State", "Region x sector"))
        sel_sectors = st.multiselect("Sectors", sectors, default=sectors[:5], key="p1_sectors", disabled=level not in ("Sector", "Subsector", "Region x sector"))
    sub = F[F.level == level]
    if level in ("Sector", "Region x sector"): sub = sub[sub.name.isin(sel_sectors)]
    if level == "Subsector": sub = sub[sub.name.str.split(" / ").str[0].isin(sel_sectors)]
    if level in ("Region", "State", "Region x sector"): sub = sub[sub.region.isin(sel_regions)]
    sub = with_measure(sub, METRIC[metric], measure); sub = sub[(sub.date >= t0) & (sub.date <= t1)].dropna(subset=["value"])
    sub["series"] = np.where(sub.level == "Region x sector", sub.region + " | " + sub.name, sub.name)
    if level == "State" and sub.series.nunique() > 12:
        top = sub.groupby("series").value.last().abs().sort_values(ascending=False).index[:12]; sub = sub[sub.series.isin(top)]; st.caption("Showing the 12 states with the largest latest values; narrow the regions to see others.")
    if len(sub):
        fig = px.line(sub, x="date", y="value", color="series", labels={"value": f"{metric} ({measure})", "date": ""}, title=f"{metric}: {measure} by {level.lower()}")
        if measure != "Index": fig.add_hline(y=0, line_color="grey", line_width=1)
        st.plotly_chart(fig, width=W)
    else: empty_note()
    st.subheader("Cross-section: states and sectors")
    bm = st.radio("Measure for the map, rankings and heat map", ["YoY %", "MoM %"], horizontal=True, key="p1_bmeasure")
    mlist = [m for m in months if pd.Timestamp(m) >= pd.Timestamp(months[0]) + pd.DateOffset(months=12 if bm == "YoY %" else 1)]
    pick = st.select_slider(f"Month for the map and rankings ({bm})", options=mlist, value=mlist[-1], key=f"p1_month_{bm}")
    c1, c2 = st.columns([1.3, 1])
    stt = with_measure(F[F.level == "State"], METRIC[metric], bm); stt = stt[stt.date == pd.Timestamp(pick)].dropna(subset=["value"])
    if len(stt):
        c1.plotly_chart(px.choropleth(stt, locations="name", locationmode="USA-states", color="value", scope="usa", color_continuous_scale="RdBu", color_continuous_midpoint=0, hover_data={"region": True},
                                      labels={"value": bm}, title=f"{metric}, {bm} by state, {pick}"), width=W)
    sec = with_measure(F[F.level == "Sector"], METRIC[metric], bm); sec = sec[sec.date == pd.Timestamp(pick)].dropna(subset=["value"]).sort_values("value")
    if len(sec):
        c2.plotly_chart(px.bar(sec, x="value", y="name", orientation="h", color="value", color_continuous_scale="RdBu", color_continuous_midpoint=0, labels={"value": bm, "name": ""}, title=f"Sectors ranked by {metric.split(' ')[0].lower()} {bm}, {pick}"), width=W)
    hm = with_measure(F[F.level == "Sector"], METRIC[metric], bm); hm = hm[(hm.date >= t0) & (hm.date <= t1)].dropna(subset=["value"])
    if len(hm):
        hm = hm.pivot_table(index="name", columns=hm.date.dt.strftime("%Y-%m"), values="value")
        st.plotly_chart(px.imshow(hm, color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto", labels={"color": bm, "x": "", "y": ""}, title=f"{metric} {bm} by sector and month", zmin=-np.nanpercentile(np.abs(hm.values), 95), zmax=np.nanpercentile(np.abs(hm.values), 95)), width=W)
    else: empty_note("YoY needs 12 months of history before the first month shown.")

# ================================================================== page 2
elif page.startswith("2"):
    st.title("Inflation forecasts: CPI, core CPI, PCE, core PCE")
    I = D["inf"]; oos = I[I.model != "first print (history)"]
    with st.expander("Forecast uncertainty settings", expanded=False):
        c1, c2 = st.columns(2)
        unc_opts = sorted(oos.model.unique()); unc_default = next((m for m in ("Ridge [B]", "Online-selected") if m in unc_opts), unc_opts[0])
        unc_model = c1.selectbox("Uncertainty from the out-of-sample errors of", unc_opts, index=unc_opts.index(unc_default), key="p2_unc")
        level = c2.select_slider("Interval", options=[80, 90, 95], value=90, key="p2_level", format_func=lambda v: f"{v}%")
    def errors(target, freq, model=None):
        g = oos[(oos.target == target) & (oos.model == (model or unc_model))]
        return (g.actual - g.forecast).dropna() if freq == "MoM" else (g.actual_yoy - g.forecast_yoy).dropna()
    zq = Z[level]
    if len(CUR):
        st.subheader(f"Current forecasts ({CUR.month.iloc[0]})")
        st.caption("Point = average of the refitted models. Intervals and probabilities use the out-of-sample errors of the model chosen in the settings above (normal approximation). "
                   "Status 'published' means the month is already out and the forecast was made from earlier months only.")
        if meta.get("kalshi_sources"):
            st.caption("Kalshi values: " + "; ".join(f"**{k}**: {v}" for k, v in meta["kalshi_sources"].items()) + ". Kalshi lists no headline PCE market, so that value is derived.")
        cols = st.columns(len(CUR))
        for c, (_, r) in zip(cols, CUR.iterrows()):
            with c.container(border=True):
                sm_, sy_ = errors(r.target, "MoM").std(), errors(r.target, "YoY").std()
                st.markdown(f"**{r.target}**  \n<span style='color:grey'>{r.get('status', '')}</span>", unsafe_allow_html=True)
                st.metric("MoM", f"{r['model average']:.2f}%", f"{zq:.0f}σ... " if False else f"{level}%: {r['model average'] - zq * sm_:.2f} to {r['model average'] + zq * sm_:.2f}", delta_color="off")
                if r.get("model average YoY") is not None:
                    st.metric("YoY", f"{r['model average YoY']:.2f}%", f"{level}%: {r['model average YoY'] - zq * sy_:.2f} to {r['model average YoY'] + zq * sy_:.2f}", delta_color="off")
                ref = [f"nowcast {r['Cleveland nowcast']:.2f} MoM"] + ([f"Kalshi {r['Kalshi']:.2f}"] if pd.notna(r.get("Kalshi")) else [])
                st.caption(" | ".join(ref))
        st.subheader("Over / under probabilities")
        c1, c2, c3 = st.columns(3)
        tgt = c1.selectbox("Target", list(CUR.target), key="ou_target"); freq = c2.radio("Measure", ["MoM", "YoY"], horizontal=True, key="ou_freq")
        row = CUR[CUR.target == tgt].iloc[0]; point = row["model average"] if freq == "MoM" else row.get("model average YoY")
        e = errors(tgt, freq)
        if point is None or len(e) < 5: empty_note("Not enough out-of-sample errors for this target and measure.")
        else:
            sd = e.std(); thr = c3.number_input("Threshold K (%)", value=float(round(point, 1)), step=0.1, format="%.2f", key=f"ou_thr_{tgt}_{freq}")
            p_norm = 1 - float(ncdf((thr - point) / sd)); p_emp = float(((point + e.values) > thr).mean())
            m1, m2, m3 = st.columns(3); m1.metric(f"P({tgt} {freq} > {thr:.2f})", f"{p_norm:.0%}", "normal approximation", delta_color="off")
            m2.metric("Empirical", f"{p_emp:.0%}", f"from {len(e)} past errors", delta_color="off"); m3.metric(f"P(≤ {thr:.2f})", f"{1 - p_norm:.0%}", delta_color="off")
            xs = np.linspace(point - 4 * sd, point + 4 * sd, 300); dens = np.exp(-0.5 * ((xs - point) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            fig = go.Figure(go.Scatter(x=xs, y=dens, line=dict(color="#4c78a8"), name="forecast distribution"))
            fig.add_trace(go.Scatter(x=xs[xs > thr], y=dens[xs > thr], fill="tozeroy", line=dict(width=0), fillcolor="rgba(231,111,81,0.45)", name=f"over {thr:.2f}"))
            fig.add_vline(x=point, line_dash="dash", annotation_text=f"forecast {point:.2f}"); fig.update_layout(height=280, title=f"{tgt} {freq}: forecast distribution", xaxis_title="%", yaxis_visible=False, margin=dict(t=50, b=30))
            c1, c2 = st.columns([1.6, 1]); c1.plotly_chart(fig, width=W)
            ks = np.round(np.arange(np.floor((point - 3 * sd) * 10) / 10, point + 3 * sd, 0.1), 1)
            ladder = pd.DataFrame({"strike K": ks, "P(> K), normal": 1 - ncdf((ks - point) / sd), "P(> K), empirical": [((point + e.values) > k).mean() for k in ks]})
            c2.dataframe(ladder.style.format({"strike K": "{:.1f}", "P(> K), normal": "{:.0%}", "P(> K), empirical": "{:.0%}"}), width=W, height=280, hide_index=True)
            st.caption("The ladder has the same format as Kalshi's 'above K' contracts: compare it with the market's prices to see where the model and the market disagree.")
    st.subheader("Model history")
    c1, c2 = st.columns([1.4, 1])
    target = c1.radio("Target", ["CPI", "core CPI", "PCE", "core PCE"], horizontal=True, key="p2_target"); freq2 = c2.radio("Measure", ["MoM", "YoY"], horizontal=True, key="p2_freq")
    fcol, acol = ("forecast", "actual") if freq2 == "MoM" else ("forecast_yoy", "actual_yoy")
    sc_rows = {}
    for m, g in oos[oos.target == target].groupby("model"):
        e_, ref_ = (g[acol] - g[fcol]).dropna(), oos[(oos.target == target) & (oos.model == "Nowcast")]
        e0 = (ref_[acol] - ref_[fcol]).dropna()
        if len(e_) < 5: continue
        a_ = g.loc[e_.index, acol]
        sc_rows[m] = {"months": len(e_), "RMSE": np.sqrt((e_ ** 2).mean()), "MAE": e_.abs().mean(), "R2": 1 - (e_ ** 2).sum() / ((a_ - a_.mean()) ** 2).sum(),
                      "RMSE / nowcast": np.sqrt((e_ ** 2).mean()) / np.sqrt((e0 ** 2).mean()) if len(e0) else np.nan, "hits 0.1 bucket": (g.loc[e_.index, fcol].round(1) == a_.round(1)).mean()}
    sc = pd.DataFrame(sc_rows).T.sort_values("RMSE") if sc_rows else pd.DataFrame()
    avail = sorted(oos[oos.target == target].model.unique())
    default = [m for m in ["Online-selected", "Nowcast", "Kalshi implied"] if m in avail] + [m for m in sc.index if m not in ("Nowcast", "Kalshi implied", "Last month", "Online-selected")][:1]
    chosen = st.multiselect("Models (the first one gets an interval band)", avail, default=default, key="p2_models")
    d = oos[(oos.target == target) & oos.model.isin(chosen) & (oos.date >= t0) & (oos.date <= t1)].dropna(subset=[fcol])
    hist = I[(I.target == target) & (I.model == "first print (history)") & (I.date >= t0) & (I.date <= t1)]
    if not st.checkbox("Show the full first-print history", value=False, key="p2_hist") and len(d): hist = hist[hist.date.between(d.date.min(), d.date.max())]
    fig = go.Figure()
    if chosen and len(d):
        g0 = d[d.model == chosen[0]].sort_values("date"); s0 = (oos[(oos.target == target) & (oos.model == chosen[0])][acol] - oos[(oos.target == target) & (oos.model == chosen[0])][fcol]).std()
        if len(g0) and np.isfinite(s0):
            fig.add_trace(go.Scatter(x=np.r_[g0.date, g0.date[::-1]], y=np.r_[g0[fcol] + zq * s0, (g0[fcol] - zq * s0)[::-1]], fill="toself", fillcolor="rgba(42,157,143,0.15)", line=dict(width=0), name=f"{chosen[0]} {level}% band"))
    fig.add_trace(go.Scatter(x=hist.date, y=hist[acol], mode="lines+markers", name="published", line=dict(color="black", width=2.5)))
    for m, g in d.groupby("model"): fig.add_trace(go.Scatter(x=g.date, y=g[fcol], mode="lines+markers", name=m, line=dict(dash="dot" if m in ("Nowcast", "Kalshi implied") else "solid")))
    fig.update_layout(title=f"{target} {freq2}: published vs forecasts (out of sample)", yaxis_title=f"% {'month on month' if freq2 == 'MoM' else 'year on year'}", height=440); st.plotly_chart(fig, width=W)
    c1, c2 = st.columns([1.3, 1])
    if len(sc):
        c1.dataframe(sc.style.format(precision=3).background_gradient(subset=["RMSE / nowcast"], cmap="RdYlGn_r"), width=W, height=380)
        top = sc.head(15).reset_index().rename(columns={"index": "model"})
        fig = px.bar(top, x="RMSE / nowcast", y="model", orientation="h", color="RMSE / nowcast", color_continuous_scale="RdYlGn_r", range_color=(0.6, 1.4), title=f"{freq2} RMSE relative to the nowcast (below 1 beats it)")
        fig.add_vline(x=1, line_dash="dash"); c2.plotly_chart(fig, width=W)
    if len(d):
        st.plotly_chart(px.bar(d.assign(error=d[acol] - d[fcol]), x="date", y="error", color="model", barmode="group", title=f"{freq2} forecast errors (published minus forecast)"), width=W)
    st.subheader("FSBI trends against this target")
    F = D["fsbi"]; c1, c2, c3, c4 = st.columns(4)
    lvl = c1.selectbox("FSBI level", ["National", "Sector", "Subsector", "Region"], key="p2_lvl"); names = sorted(F.loc[F.level == lvl, "name"].unique())
    nm = c2.selectbox("Series", names, key="p2_name"); met = c3.selectbox("FSBI metric", ["ticket_index", "txn_index", "sales_index"], format_func=lambda x: x.split("_")[0], key="p2_met")
    lag = c4.slider("FSBI leads by (months)", 0, 3, 0, key="p2_lag")
    fs_all = with_measure(F[(F.level == lvl) & (F.name == nm)], met, "MoM %" if freq2 == "MoM" else "YoY %").set_index("date")["value"]
    tg_all = I[(I.target == target) & (I.model == "first print (history)")].set_index("date")[acol]
    both = pd.concat([fs_all.shift(lag).rename("fsbi"), tg_all.rename("target")], axis=1).dropna(); both = both[(both.index >= t0) & (both.index <= t1)]
    if len(both) > 5:
        fig = go.Figure(); fig.add_trace(go.Scatter(x=both.index, y=both.target, name=f"{target} {freq2} (%)", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=both.index, y=both.fsbi, name=f"{nm} {met.split('_')[0]} {freq2} % (lag {lag})", yaxis="y2", line=dict(color="#e76f51")))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title=f"FSBI {freq2} %"), yaxis_title=f"{target} {freq2} %", height=380, title=f"r = {both.corr().iloc[0, 1]:+.2f} over {len(both)} months")
        c1, c2 = st.columns([2.2, 1]); c1.plotly_chart(fig, width=W)
        lc = pd.DataFrame({"lag": range(4), "r": [pd.concat([fs_all.shift(l), tg_all], axis=1).dropna().loc[t0:t1].corr().iloc[0, 1] for l in range(4)]})
        c2.plotly_chart(px.bar(lc, x="lag", y="r", title="Correlation by lead (months)", range_y=(-1, 1)).update_layout(height=380), width=W)
    else: empty_note("Too few overlapping months for this FSBI series and target in the timeline.")

# ================================================================== page 3
else:
    st.title("Trading strategies")
    T, G = D["strat"], D["sig"]; latest = meta.get("latest", {})
    cpi_now = CUR.loc[CUR.target == "CPI"].iloc[0] if len(CUR) and (CUR.target == "CPI").any() else None
    DESCR = [("FSBI sector: demand rotation", "Ranks sector ETFs each FSBI day by their FSBI sectors' **transaction** growth (z-score vs earlier months); long the top third, short the bottom third, to month-end.", "sector volume (or acceleration / ticket)", "sector ETFs, long/short"),
             ("FSBI sector: healthy vs squeeze", "Long sectors whose FSBI volume and ticket are both above normal; short sectors raising prices (ticket up) on falling volume.", "sector volume and ticket", "sector ETFs"),
             ("FSBI sector: pricing-power", "Long sectors whose FSBI ticket predicts the hottest CPI-component inflation vs its recent run-rate, short the coldest, into the CPI print.", "sector ticket -> CPI component", "sector ETFs"),
             ("FSBI sector: regional breadth", "Cyclicals (XLY) over defensives (XLP) when 3-4 regions' small-business sales accelerate; the reverse when 0-1 do.", "regional sales growth", "XLY vs XLP"),
             ("FSBI PCA", "Demand rotation on each sector's **idiosyncratic** FSBI volume: the part left after removing common consumer factors (rolling PCA on earlier months).", "sector volume minus common factors", "sector ETFs, long/short"),
             ("XLP/XLY", "Defensive staples (XLP) when the CPI forecast is high or hotter than a reference, cyclicals (XLY) otherwise.", "CPI forecast (FSBI + macro + Kalshi models)", "XLP or XLY"),
             ("TIPS", "Long short-term TIPS (STIP) when the CPI forecast beats the monthly breakeven; the breakeven version shorts SHY against it.", "CPI forecast vs breakeven", "STIP (and SHY)"),
             ("Large-small", "Long Russell 1000, short Russell 2000 when the signal says inflation pressure (scale wins, customers substitute toward large firms).", "inflation expectation or FSBI squeeze", "IWB vs IWM"),
             ("Single-name:", "Single stocks chosen by FSBI sector signals: commodity producers on the matching sector ticket, or large vs small operators within a sector on the sector's FSBI squeeze. Today's listings: survivorship bias.", "sector ticket / squeeze", "single stocks"),
             ("Release gap", "Overnight position from the close before the CPI release to the release-morning open, in the direction implied by the model's predicted market surprise.", "predicted CPI surprise", "SPY, QQQ, IWM, TLT"),
             ("Kalshi", "Kalshi CPI / PCE contracts: bull spreads or single contracts around the model forecast, or drift trades exited before the release. P&L in $ per contract.", "forecast vs the market ladder", "Kalshi"),
             ("Industry contract", "Product-price direction contracts against an assumed market (no historical quotes): an upper bound on the edge.", "product-matched sector ticket", "Kalshi (assumed)")]
    describe = lambda n: next((x for x in DESCR if n.startswith(x[0])), ("", "See the notebook for details.", "", ""))
    c1, c2 = st.columns([1, 2.2])
    fams = sorted(T.family.unique()); fam = c1.selectbox("Strategy family", fams, index=fams.index("FSBI sector strategies") if "FSBI sector strategies" in fams else 0, key="p3_family")
    pool = T[T.family == fam]; rk = pool.groupby("strategy").ret.apply(lambda r: r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else -9).sort_values(ascending=False)
    strat = c2.selectbox("Strategy (sorted by Sharpe)", list(rk.index), format_func=lambda n: f"{n}   (Sharpe {rk[n]:.2f})", key="p3_strategy")
    S0 = pool[pool.strategy == strat].set_index("date").sort_index(); dollar = S0.unit.iloc[0] == "$"; r_all = S0.ret; r_win = r_all[(r_all.index >= t0) & (r_all.index <= t1)]
    _, what, fsbi_in, instr = describe(strat)
    with st.container(border=True):
        st.markdown(f"#### {strat}\n{what}\n\n**FSBI input:** {fsbi_in or 'n/a'} | **Instrument:** {instr or 'n/a'} | **Months in the timeline:** {len(r_win)} of {len(r_all)}")
        m = metrics(r_win, dollar); mc = st.columns(6)
        items = [("Sharpe", f"{m.get('Sharpe', np.nan):.2f}"), ("Sortino", f"{m.get('Sortino', np.nan):.2f}"), ("win rate", f"{m.get('win rate', np.nan):.0%}")]
        items += [("total $ / contract", f"{m.get('total $ / contract', np.nan):.2f}"), ("mean $ / month", f"{m.get('mean $ / month', np.nan):.3f}"), ("months", f"{m.get('months', 0)}")] if dollar else \
                 [("ROI", f"{m.get('ROI %', np.nan):+.1f}%"), ("ann. vol", f"{m.get('ann. vol %', np.nan):.1f}%"), ("max drawdown", f"{m.get('max drawdown %', np.nan):.1f}%")]
        for c, (k, v) in zip(mc, items): c.metric(k, v)

    st.header("1. Backtest")
    compare = st.multiselect("Compare with", [s for s in T.strategy.unique() if s != strat and (T.loc[T.strategy == s, "unit"].iloc[0] == "$") == dollar], key=f"cmp_{kid(strat)}")
    cmp = T[T.strategy.isin([strat] + compare) & (T.date >= t0) & (T.date <= t1)].sort_values("date")
    if len(cmp):
        if dollar:
            cmp = cmp.assign(cum=cmp.groupby("strategy").ret.cumsum()); st.plotly_chart(px.line(cmp, x="date", y="cum", color="strategy", markers=True, title="Cumulative $ per contract").add_hline(y=0, line_color="grey"), width=W)
        else:
            cmp = cmp.assign(growth=cmp.groupby("strategy").ret.transform(lambda r: (1 + r).cumprod())); c1, c2 = st.columns(2)
            c1.plotly_chart(px.line(cmp, x="date", y="growth", color="strategy", title="Growth of $1 (net of costs)").add_hline(y=1, line_color="grey").update_layout(legend=dict(orientation="h", y=-0.25)), width=W)
            dd = cmp.assign(drawdown=cmp.groupby("strategy").growth.transform(lambda c: 100 * (c / c.cummax() - 1)))
            c2.plotly_chart(px.area(dd, x="date", y="drawdown", color="strategy", title="Drawdown (%)").update_layout(legend=dict(orientation="h", y=-0.25)), width=W)
        st.plotly_chart(px.bar(x=r_win.index, y=r_win.values * (1 if dollar else 100), color=np.where(r_win.values > 0, "gain", "loss"), color_discrete_map={"gain": "#2a9d8f", "loss": "#e76f51"},
                               labels={"x": "", "y": "$" if dollar else "%"}, title="Monthly result"), width=W)
    else: empty_note()

    st.header("2. Strategy lab")
    k_ = kid(strat); lab = {"ret": None, "pos": None, "now_pos": None, "now_text": None, "bench": None, "pos_label": None}
    def fc_now(src, shift):
        if cpi_now is None: return np.nan
        return {"model forecast": cpi_now["model average"] + shift, "Cleveland nowcast": cpi_now["Cleveland nowcast"], "Kalshi": cpi_now.get("Kalshi"), "perfect foresight (actual)": np.nan}.get(src, np.nan)
    if strat.startswith(("XLP/XLY", "TIPS")):
        tips = strat.startswith("TIPS"); c1, c2, c3 = st.columns(3)
        src = c1.selectbox("Forecast used", ["model forecast", "Cleveland nowcast", "Kalshi", "perfect foresight (actual)"], key=f"src_{k_}")
        fc = {"model forecast": G.forecast_cpi, "Cleveland nowcast": G.nowcast_cpi, "Kalshi": G.kalshi_cpi, "perfect foresight (actual)": G.actual_cpi}[src]
        win = c2.radio("Hold", ["to month-end", "through release"], index=1 if "through release" in strat else 0, horizontal=True, key=f"win_{k_}"); w = "me" if win == "to month-end" else "rel"
        cost = c3.slider("Cost per side (bp)", 0.0, 20.0, 5.0, 0.5, key=f"cost_{k_}") / 1e4
        shift = st.slider("Scenario: shift today's CPI forecast (pp; negative = optimistic, positive = pessimistic)", -0.3, 0.3, 0.0, 0.05, key=f"shift_{k_}")
        now_f = fc_now(src, shift)
        if not tips:
            refs = ["|forecast| above threshold", "forecast above nowcast + threshold", "forecast above Kalshi + threshold"]
            ref_ = st.selectbox("Signal", refs, index=2 if "Kalshi" in strat else 1 if "nowcast" in strat else 0, key=f"ref_{k_}")
            thr = st.slider("Threshold (pp)", 0.0, 0.5, 0.2 if ref_.startswith("|") else 0.0, 0.01, key=f"thr_{k_}_{refs.index(ref_)}")
            hot = (fc.abs() > thr) if ref_.startswith("|") else (fc > G.nowcast_cpi + thr) if "nowcast" in ref_ else (fc > G.kalshi_cpi + thr)
            ok = fc.notna() & (G.kalshi_cpi.notna() if "Kalshi" in ref_ else True)
            pos = pd.Series(np.where(hot, 1, -1), index=G.index).where(ok); ret = pd.Series(np.where(pos > 0, G[f"r_XLP_{w}"], G[f"r_XLY_{w}"]), index=G.index).where(ok) - 2 * cost
            bench = (G[f"r_XLY_{w}"] - 2 * cost).where(ok).rename("always XLY"); lab["pos_label"] = {1: "XLP (defensive)", -1: "XLY (cyclical)"}
            if np.isfinite(now_f):
                refv = None if ref_.startswith("|") else (cpi_now["Cleveland nowcast"] if "nowcast" in ref_ else cpi_now.get("Kalshi"))
                if ref_.startswith("|") or (refv is not None and pd.notna(refv)):
                    lab["now_pos"] = 1 if (abs(now_f) > thr if ref_.startswith("|") else now_f > refv + thr) else -1
        else:
            mode = st.selectbox("Mode", ["long-only (STIP or cash)", "long/short vs SHY"], index=1 if "breakeven" in strat else 0, key=f"mode_{k_}")
            thr = st.slider("Threshold over the monthly breakeven (pp)", -0.2, 0.4, 0.1, 0.01, key=f"thr_{k_}")
            dlt = fc - G.breakeven_monthly; ok = dlt.notna()
            if mode.startswith("long-only"):
                pos = pd.Series(np.where(dlt > thr, 1, 0), index=G.index).where(ok); ret = (pos * (G[f"r_STIP_{w}"] - 2 * cost)).where(ok)
            else:
                pos = pd.Series(np.select([dlt > thr, dlt < -thr], [1, -1], 0), index=G.index).where(ok); ret = (pos * (G[f"r_STIP_{w}"] - 1.44 * G[f"r_SHY_{w}"]) - 2 * cost * pos.abs() * 2.44).where(ok)
            bench = (G[f"r_STIP_{w}"] - 2 * cost).where(ok).rename("always STIP"); lab["pos_label"] = {1: "long TIPS", 0: "flat", -1: "short TIPS vs SHY"}
            be_now = G.breakeven_monthly.dropna().iloc[-1] if G.breakeven_monthly.notna().any() else np.nan
            if np.isfinite(now_f) and np.isfinite(be_now):
                d_ = now_f - be_now; lab["now_pos"] = (1 if d_ > thr else 0) if mode.startswith("long-only") else (1 if d_ > thr else -1 if d_ < -thr else 0)
        lab.update(ret=ret, pos=pos, bench=bench)
        if np.isfinite(now_f): lab["now_text"] = f"Today's CPI forecast ({src}{', shifted' if shift and src == 'model forecast' else ''}): **{now_f:.2f}% MoM**"
    elif strat.startswith("Large-small") and any(c.startswith("r_size_") for c in G):
        names_ = {"z_fsbi_squeeze": "FSBI squeeze (ticket - transactions)", "z_sector_squeeze": "FSBI sector squeeze breadth", "z_kalshi": "Kalshi implied CPI", "z_forecast": "model CPI forecast", "z_surprise": "expected surprise (forecast - nowcast)"}
        opts = [c for c in names_ if c in G]; guess = next((c for c, lbl in names_.items() if c in G and lbl.split(" (")[0].lower() in strat.lower()), opts[0])
        guess = "z_sector_squeeze" if "breadth" in strat and "z_sector_squeeze" in G else guess
        c1, c2, c3, c4 = st.columns(4)
        sname = c1.selectbox("Signal (z-score vs earlier months)", opts, index=opts.index(guess), format_func=names_.get, key=f"sig_{k_}")
        hs = [h for h in (5, 21, 63) if f"r_size_{h}d" in G]; hz = c2.selectbox("Horizon (trading days)", hs, index=hs.index(21) if 21 in hs else 0, key=f"h_{k_}")
        thr = c3.slider("Trade only when |z| exceeds", 0.0, 2.0, 0.0, 0.1, key=f"thr_{k_}"); cost = c4.slider("Cost per side (bp)", 0.0, 20.0, 5.0, 0.5, key=f"cost_{k_}") / 1e4
        z = G[sname]; ok = z.notna() & G[f"r_size_{hz}d"].notna()
        pos = pd.Series(np.select([z > thr, z < -thr], [1, -1], 0), index=G.index).where(ok); ret = (pos * G[f"r_size_{hz}d"] - 4 * cost * pos.abs()).where(ok)
        lab.update(ret=ret, pos=pos, bench=(G[f"r_size_{hz}d"] - 4 * cost).where(ok).rename("always long large / short small"), pos_label={1: "long large / short small", 0: "flat", -1: "long small / short large"})
        zn = latest.get(sname)
        if zn is not None: lab["now_pos"] = 1 if zn > thr else -1 if zn < -thr else 0; lab["now_text"] = f"Latest {names_[sname]} z-score ({latest.get('month')}): **{zn:+.2f}**"
    elif strat.startswith("FSBI sector: regional breadth") and "breadth_n" in G:
        c1, c2, c3 = st.columns(3)
        broad = c1.slider("Broad if at least this many regions accelerate", 2, 4, 3, key=f"b_{k_}"); narrow = c2.slider("Narrow if at most", 0, 2, 1, key=f"n_{k_}"); cost = c3.slider("Cost per side (bp)", 0.0, 20.0, 5.0, 0.5, key=f"cost_{k_}") / 1e4
        b = G.breadth_n; ok = b.notna(); pos = pd.Series(np.select([b >= broad, b <= narrow], [1, -1], 0), index=G.index).where(ok)
        lab.update(ret=(pos * (G.r_XLY_me - G.r_XLP_me) - 4 * cost * pos.abs()).where(ok), pos=pos, bench=(G.r_XLY_me - G.r_XLP_me - 4 * cost).where(ok).rename("always XLY - XLP"), pos_label={1: "long XLY / short XLP", 0: "flat", -1: "long XLP / short XLY"})
        bn = latest.get("breadth_n")
        if bn is not None: lab["now_pos"] = 1 if bn >= broad else -1 if bn <= narrow else 0; lab["now_text"] = f"Regions accelerating in the latest month ({latest.get('month')}): **{int(bn)} of 4**"
    elif strat.startswith(("FSBI sector:", "FSBI PCA")) and "+ " not in strat and len(D["secsig"]):
        SS, SR = D["secsig"], D["secret"]; sigs = sorted(SS.signal.unique())
        guess = "idiosyncratic (PCA)" if "PCA" in strat else "pricing-power heat" if "pricing" in strat else "volume acceleration" if "acceleration" in strat else "ticket" if "by ticket" in strat else "volume"
        c1, c2, c3 = st.columns(3)
        modes = ["long-short", "long-only vs equal-weight universe", "healthy vs squeeze (volume and ticket)"]
        mode = c1.selectbox("Construction", modes, index=2 if "healthy" in strat else 0, key=f"mode_{k_}")
        sname = c2.selectbox("Ranking signal", sigs, index=sigs.index(guess) if guess in sigs else 0, key=f"sig_{k_}", disabled=mode.startswith("healthy"))
        win = c3.radio("Hold", ["to month-end", "through release"], index=1 if "through release" in strat else 0, horizontal=True, key=f"win_{k_}"); w = "me" if win == "to month-end" else "rel"
        c1, c2 = st.columns(2); frac = c1.slider("Share of ETFs in each leg", 0.2, 0.5, 1 / 3, 0.05, key=f"frac_{k_}", disabled=mode.startswith("healthy")); cost = c2.slider("Cost per side (bp)", 0.0, 20.0, 5.0, 0.5, key=f"cost_{k_}") / 1e4
        R = SR[SR.window == w].pivot_table(index="date", columns="etf", values="ret"); out, now_long, now_short = {}, [], []
        if mode.startswith("healthy"):
            V, P = (SS[SS.signal == s].pivot_table(index="date", columns="etf", values="value") for s in ("volume", "ticket"))
            for t in R.index.intersection(V.index):
                v, p_, r_ = V.loc[t], P.reindex(V.index).loc[t], R.loc[t]; okm = v.notna() & p_.notna() & r_.reindex(v.index).notna()
                if okm.sum() < 4: continue
                ps = pd.Series(0.0, index=v.index[okm]); ps[(v[okm] > 0) & (p_[okm] > 0)] = 1; ps[(v[okm] < 0) & (p_[okm] > 0)] = -1; n = (ps != 0).sum()
                out[t] = (ps * r_.reindex(ps.index)).sum() / n - 2 * cost if n else 0.0
            vl, pl = V.iloc[-1], P.reindex(V.index).iloc[-1]; now_long = list(vl.index[(vl > 0) & (pl > 0)]); now_short = list(vl.index[(vl < 0) & (pl > 0)])
        else:
            Sg = SS[SS.signal == sname].pivot_table(index="date", columns="etf", values="value")
            for t in R.index.intersection(Sg.index):
                s_ = Sg.loc[t].dropna(); r_ = R.loc[t].reindex(s_.index).dropna(); s_ = s_[r_.index]
                if len(s_) < 4: continue
                n = max(1, int(round(len(s_) * frac))); o = s_.sort_values(kind="stable")
                out[t] = (r_[o.index[-n:]].mean() - r_[o.index[:n]].mean() - 4 * cost) if mode == "long-short" else (r_[o.index[-n:]].mean() - r_.mean() - 2 * cost)
            last = Sg.dropna(how="all").iloc[-1].dropna().sort_values(); n = max(1, int(round(len(last) * frac))); now_long, now_short = list(last.index[-n:]), (list(last.index[:n]) if mode == "long-short" else [])
        lab["ret"] = pd.Series(out, dtype=float)
        lab["now_text"] = f"Positions from the latest FSBI month: long **{', '.join(now_long) or 'none'}**" + (f"; short **{', '.join(now_short)}**" if now_short else "")
        if len(D["secmap"]):
            with st.expander("Which FSBI sectors feed each ETF"):
                st.dataframe(D["secmap"].dropna(subset=["etf"])[["entity", "group", "etf", "cpi_component"]], width=W, hide_index=True)
    elif strat.startswith("Single-name") and len(D["single"]):
        SN = D["single"]; base = strat.replace("Single-name: ", ""); nm = next((n for n in SN.name.unique() if base.startswith(n)), SN.name.unique()[0])
        c1, c2, c3, c4 = st.columns(4)
        nm = c1.selectbox("Basket", list(SN.name.unique()), index=list(SN.name.unique()).index(nm), key=f"bk_{k_}"); sub = SN[SN.name == nm]
        hz = c2.selectbox("Horizon", list(sub.horizon.astype(str).unique()), key=f"h_{k_}"); sub = sub[sub.horizon.astype(str) == hz].set_index("date")
        sigc = c3.selectbox("Signal", [c for c in ("z_squeeze", "z_combo") if sub[c].notna().any()], format_func=lambda c: {"z_squeeze": "FSBI sector signal", "z_combo": "FSBI squeeze + Kalshi CPI"}[c], index=1 if "Kalshi" in strat and sub.z_combo.notna().any() else 0, key=f"sig_{k_}")
        thr = c4.slider("Trade only when |z| exceeds", 0.0, 2.0, 0.0, 0.1, key=f"thr_{k_}"); flip = st.checkbox("Reverse the hypothesis", value=False, key=f"flip_{k_}"); cost = 0.0005
        z = sub[sigc] * (-1 if flip else 1); ok = z.notna() & sub.ret.notna(); pos = pd.Series(np.select([z > thr, z < -thr], [1, -1], 0), index=sub.index).where(ok)
        lab.update(ret=(pos * sub.ret - 4 * cost * pos.abs()).where(ok), pos=pos, pos_label={1: "long the hypothesis basket", 0: "flat", -1: "the reverse"})
        zn = (latest.get("single_name", {}).get(nm) or {}).get(sigc)
        if zn is not None: zn *= (-1 if flip else 1); lab["now_pos"] = 1 if zn > thr else -1 if zn < -thr else 0; lab["now_text"] = f"Latest signal ({latest.get('month')}): **z = {zn:+.2f}**"
    if lab["ret"] is None:
        st.caption("This strategy's raw ingredients are not exported (or it is an overlay), so the lab adjusts its monthly results directly.")
        c1, c2, c3, c4, c5 = st.columns(5)
        flip = c1.checkbox("Reverse the trade", value=False, key=f"gflip_{k_}"); scale = c2.slider("Position size (x)", 0.25, 3.0, 1.0, 0.25, key=f"gscale_{k_}")
        extra = c3.slider("Extra cost per month (" + ("$" if dollar else "bp") + ")", 0.0, 0.05 if dollar else 20.0, 0.0, 0.005 if dollar else 0.5, key=f"gcost_{k_}")
        vt = c4.slider("Vol target (annual %, 0 = off)", 0, 30, 0, 1, key=f"gvt_{k_}", disabled=dollar); cap = c5.slider("Monthly loss cap (%, 0 = off)", 0, 20, 0, 1, key=f"gcap_{k_}", disabled=dollar)
        r = r_all * (-1 if flip else 1) * scale - (extra if dollar else extra / 1e4)
        if vt and not dollar:
            trail = r_all.rolling(12, min_periods=6).std().shift(1) * np.sqrt(12); r = r * (vt / 100 / trail).clip(upper=3).fillna(1.0)
        if cap and not dollar: r = r.clip(lower=-cap / 100); st.caption("The loss cap assumes a stop could have been filled at that level within the month (an approximation on monthly data).")
        lab["ret"] = r
    ret = lab["ret"].dropna(); ret = ret[(ret.index >= t0) & (ret.index <= t1)]
    if len(ret) < 3: st.warning("Too few months in the timeline for this setting.")
    else:
        curve = pd.DataFrame({"lab version": ret.cumsum() if dollar else (1 + ret).cumprod(), "as backtested": (r_win.reindex(ret.index).fillna(0).cumsum() if dollar else (1 + r_win.reindex(ret.index).fillna(0)).cumprod())})
        if lab["bench"] is not None: curve[lab["bench"].name] = (1 + lab["bench"].reindex(ret.index).fillna(0)).cumprod()
        c1, c2 = st.columns([1.6, 1])
        c1.plotly_chart(px.line(curve, title="Lab version vs the backtest" + (" and benchmark" if lab["bench"] is not None else ""), labels={"value": "$" if dollar else "growth of $1", "index": ""}).add_hline(y=0 if dollar else 1, line_color="grey"), width=W)
        mt = pd.DataFrame({"lab version": metrics(ret, dollar), "as backtested": metrics(r_win.reindex(ret.index), dollar), **({lab["bench"].name: metrics(lab["bench"].reindex(ret.index))} if lab["bench"] is not None else {})}).T
        c2.dataframe(mt.style.format(precision=2), width=W)
        if lab["pos"] is not None and lab["pos_label"]:
            pdf = pd.DataFrame({"date": lab["pos"].index, "position": lab["pos"].map(lab["pos_label"])}).dropna(); pdf = pdf[(pdf.date >= t0) & (pdf.date <= t1)]
            st.plotly_chart(px.scatter(pdf, x="date", y="position", color="position", title="Position each month").update_layout(height=260, showlegend=False), width=W)

    st.header("3. Projection")
    if lab["now_text"]: st.markdown(lab["now_text"])
    if lab["now_pos"] is not None and lab["pos_label"]: st.markdown(f"Position implied **now** under these settings: **{lab['pos_label'].get(lab['now_pos'], lab['now_pos'])}**")
    if len(ret) < 6: st.info("The projection needs at least 6 months of lab results in the timeline.")
    else:
        c1, c2, c3 = st.columns(3); horizon = c1.slider("Months ahead", 3, 24, 12, key="mc_h"); sims = c2.slider("Simulations", 500, 5000, 2000, 500, key="mc_n")
        can_cond = lab["pos"] is not None and lab["now_pos"] is not None
        cond = c3.checkbox("Resample only months with today's position", value=can_cond, disabled=not can_cond, key=f"mc_cond_{k_}")
        sample = ret
        if cond and can_cond:
            same = ret[lab["pos"].reindex(ret.index) == lab["now_pos"]]
            if len(same) >= 6: sample = same; st.caption(f"Conditioned on {len(same)} historical months with today's position.")
            else: st.caption(f"Only {len(same)} months had today's position; using all months.")
        rng = np.random.default_rng(7); vals = sample.values; n = len(vals); block = 3; paths = np.empty((sims, horizon))
        for i in range(sims):
            idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, horizon // block + 1)])[:horizon]
            paths[i] = np.cumsum(vals[idx]) if dollar else np.cumprod(1 + vals[idx]) - 1
        scale_ = 1 if dollar else 100; q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0) * scale_; x = np.arange(1, horizon + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[4], q[0][::-1]], fill="toself", fillcolor="rgba(76,120,168,0.15)", line=dict(width=0), name="5-95%"))
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[q[3], q[1][::-1]], fill="toself", fillcolor="rgba(76,120,168,0.35)", line=dict(width=0), name="25-75%"))
        fig.add_trace(go.Scatter(x=x, y=q[2], line=dict(color="#1d3557", width=2.5), name="median")); fig.add_hline(y=0, line_color="grey")
        fig.update_layout(title=f"Simulated cumulative {'$ per contract' if dollar else 'return (%)'} over the next {horizon} months ({sims} block-bootstrap paths)", xaxis_title="months ahead", height=420)
        st.plotly_chart(fig, width=W)
        end = paths[:, -1] * scale_; u = "$" if dollar else "%"; c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median outcome", f"{np.median(end):+.2f}{u}"); c2.metric("Probability of a loss", f"{(end < 0).mean():.0%}"); c3.metric("5% worst case", f"{np.percentile(end, 5):+.2f}{u}"); c4.metric("95% best case", f"{np.percentile(end, 95):+.2f}{u}")
        st.caption("Resamples the lab version's own monthly results in 3-month blocks. It assumes the future resembles the backtest: it shows the spread of outcomes, not proof of an edge.")
