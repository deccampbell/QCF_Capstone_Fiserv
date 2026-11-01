"""Exercises every page, every strategy, and checks that each chart actually contains data."""
import json, os
os.environ.setdefault("FSBI_DASH_DATA", "data/dashboard")
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
at = AppTest.from_file("app.py", default_timeout=180).run(); check(at, "p1 default (sector, YoY)", 4)
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
fams = list(at.selectbox(key="p3_family").options)
for fam in fams:
    at.selectbox(key="p3_family").set_value(fam).run()
    for strat in list(at.selectbox(key="p3_strategy").options):
        name = strat.split("   (Sharpe")[0]
        at.selectbox(key="p3_strategy").set_value(name).run(); check(at, f"p3 [{fam}] {name[:60]}", 3)
print("ALL DASHBOARD CHECKS PASSED")
