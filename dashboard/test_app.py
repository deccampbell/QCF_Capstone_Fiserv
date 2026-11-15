"""Exercises every page, every strategy, and checks that each chart actually contains data."""
import json, os
os.environ.setdefault("FSBI_DASH_DATA", "Data/processed/dashboard")
from streamlit.testing.v1 import AppTest
def chart_has_data(el):
    spec = json.loads(el.proto.spec)
    for tr in spec.get("data", []):
        for key in ("y", "z", "x"):
            v = tr.get(key)
            if isinstance(v, dict) and "bdata" in v:                      # plotly's binary encoding: decode and require a finite value
                import base64, numpy as np
                arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
                if arr.dtype.kind in "fc" and np.isfinite(arr).any(): return True
                if arr.dtype.kind in "iub" and arr.size: return True
                continue
            if isinstance(v, list) and any(x is not None and x == x for x in (v if not (v and isinstance(v[0], list)) else [y for row in v for y in row])): return True
    return False
def check(at, label, min_charts=1):
    errs = [e.value for e in at.exception]
    charts = at.get("plotly_chart"); empty = [json.loads(c.proto.spec).get("layout", {}).get("title", {}).get("text", "?") for c in charts if not chart_has_data(c)]
    status = "OK" if not errs and not empty and len(charts) >= min_charts else f"PROBLEM errors={str(errs)[:300]} empty_charts={empty} n_charts={len(charts)}"
    print(f"{label}: {status} | charts {len(charts)}, tables {len(at.dataframe)}, metrics {len(at.metric)}"); assert status == "OK", label
at = AppTest.from_file("app.py", default_timeout=180).run(); check(at, "p0 overview (default page)", 0)
rel = [d.value for d in at.dataframe if "When" in d.value.columns]
assert rel and len(rel[0]) >= 4, "release calendar missing from the overview"
print("release calendar:", rel[0][["Release", "Days left", "When"]].to_dict("records"))
md = " ".join(m.value for m in at.markdown)
assert "Last month" in md, "previous-month check missing from the forecast cards"
at.radio(key="page").set_value("1. FSBI explorer").run(); check(at, "p1 default (sector, YoY)", 4)
for lvl in ("National", "Subsector", "Region", "State", "Region x sector"):
    for ms in ("MoM %", "YoY %", "Index"):
        at.selectbox(key="p1_level").set_value(lvl); at.radio(key="p1_measure").set_value(ms).run(); check(at, f"p1 {lvl} / {ms}", 3)
at.radio(key="p1_bmeasure").set_value("MoM %").run(); check(at, "p1 bottom charts MoM", 4)
at.radio(key="page").set_value("2. Inflation forecasts").run(); check(at, "p2 default", 4)
for tg in ("CPI", "core CPI", "PCE", "core PCE"):
    for fq in ("MoM", "YoY"):
        at.radio(key="p2_target").set_value(tg); at.radio(key="p2_freq").set_value(fq).run(); check(at, f"p2 history {tg} {fq}", 4)
at.radio(key="ou_freq").set_value("YoY").run(); check(at, "p2 over/under YoY", 4)
at.selectbox(key="p2_lvl").set_value("Sector").run(); at.slider(key="p2_lag").set_value(2).run(); check(at, "p2 FSBI overlay sector lag 2", 5)
at.radio(key="page").set_value("3. Trading strategies").run(); check(at, "p3 default", 3)
if len(at.selectbox(key="lb_rank").options) > 1:
    for rb in list(at.selectbox(key="lb_rank").options)[:3]:
        at.selectbox(key="lb_rank").set_value(rb).run(); check(at, f"p3 leaderboard ranked by {rb}", 3)
    at.checkbox(key="lb_sig").check().run(); check(at, "p3 leaderboard, significant only", 2); at.checkbox(key="lb_sig").uncheck().run()
fams = list(at.selectbox(key="p3_family").options)
for fam in fams:
    at.selectbox(key="p3_family").set_value(fam).run()
    for strat in list(at.selectbox(key="p3_strategy").options):
        name = strat.split("   (Sharpe")[0]
        at.selectbox(key="p3_strategy").set_value(name).run(); check(at, f"p3 [{fam}] {name[:60]}", 3)


# ---- pages and components added in the merged dashboard
at = AppTest.from_file("app.py", default_timeout=180).run()
for pg, n in (("4. Economic applications", 3), ("5. Signal lab", 4), ("6. Findings", 0)):
    at.radio(key="page").set_value(pg).run(); check(at, pg, n)
at.radio(key="page").set_value("5. Signal lab").run()
for sig in list(at.selectbox(key="lab_signal").options)[:12]:
    at.selectbox(key="lab_signal").set_value(sig).run(); check(at, f"lab signal {sig[:50]}", 2)
for w in at.selectbox(key="lab_window").options:
    at.selectbox(key="lab_window").set_value(w).run(); check(at, f"lab window {w}", 2)
for tg in at.radio(key="rel_target").options:
    for src in at.radio(key="rel_source").options:
        at.radio(key="rel_target").set_value(tg); at.radio(key="rel_source").set_value(src).run(); check(at, f"release {tg} / {src[:20]}", 2)
for s in at.selectbox(key="xs_signal").options:
    at.selectbox(key="xs_signal").set_value(s).run(); check(at, f"xs {s[:50]}", 3)
at.radio(key="page").set_value("2. Inflation forecasts").run()
for v in list(at.selectbox(key="imp_view").options):
    at.selectbox(key="imp_view").set_value(v).run(); check(at, f"importance {v}", 4)
at.radio(key="page").set_value("3. Trading strategies").run()
for fam in list(at.selectbox(key="p3_family").options):
    at.selectbox(key="p3_family").set_value(fam).run()
    strat = at.selectbox(key="p3_strategy").options[0].split("   (Sharpe")[0]; k = __import__("re").sub(r"\W+", "_", strat)[:48]
    keys = {w.key for w in at.slider} | {w.key for w in at.radio} | {w.key for w in at.selectbox} | {w.key for w in at.checkbox}
    kind = "timing" if f"thr_{k}" in keys else "xs" if f"frac_{k}" in keys else "fixed" if f"gflip_{k}" in keys else "?"
    if kind == "timing":
        at.slider(key=f"thr_{k}").set_value(at.slider(key=f"thr_{k}").max / 2); at.radio(key=f"size_{k}").set_value("Proportional (capped at 2 sd)"); at.checkbox(key=f"flip_{k}").check().run()
    elif kind == "xs":
        at.selectbox(key=f"mode_{k}").set_value("long-only vs equal-weight universe"); at.slider(key=f"frac_{k}").set_value(0.25).run()
    elif kind == "fixed":
        at.checkbox(key=f"gflip_{k}").check(); at.slider(key=f"gvt_{k}").set_value(10); at.slider(key=f"gcap_{k}").set_value(5).run()
    check(at, f"lab [{fam}] {strat[:45]} ({kind})", 3)

# ---- design/live score periods, economic signals (leading indicators, small caps)
at = AppTest.from_file("app.py", default_timeout=180).run()
at.radio(key="page").set_value("2. Inflation forecasts").run(); check(at, "p2 live cards", 4)
for per in ("Live track record", "All", "Design period"):
    at.radio(key="p2_period").set_value(per).run(); check(at, f"p2 scores {per}", 4)
at.radio(key="page").set_value("4. Economic applications").run()
for msr in list(at.selectbox(key="lead_measure").options):
    at.selectbox(key="lead_measure").set_value(msr).run(); check(at, f"p4 leading indicators: {msr}", 3)
titles = [json.loads(c.proto.spec).get("layout", {}).get("title", {}).get("text", "") for c in at.get("plotly_chart")]
assert any("sector returns" in t for t in titles), "small-cap leading-indication chart missing"
print("ALL DASHBOARD CHECKS PASSED")
