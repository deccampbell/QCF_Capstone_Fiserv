# FSBI x Inflation dashboard

The sidebar holds what applies everywhere: the data folder, page navigation and the **timeline**. Page-specific controls sit on each page.

1. **FSBI explorer.** National KPI cards; level, metric (sales / transactions / ticket), measure (index, MoM %, YoY %), and region and sector filters; trend lines; a state map and sector ranking for any month; a sector-by-month heat map.
2. **Inflation forecasts.**
   - Current forecasts for CPI, core CPI, PCE and core PCE, **MoM and YoY**, with intervals at 80/90/95%.
   - An **over/under calculator**: P(print > K) from a normal approximation and from the empirical error distribution, a forecast-distribution chart, and a strike ladder in Kalshi's "above K" format.
   - Model history in MoM or YoY, with an interval band, error tables and an error chart.
   - An FSBI overlay with a lead slider and correlation by lead.
   - Uncertainty comes from the out-of-sample errors of the model chosen under *Forecast uncertainty settings*.
3. **Trading strategies.** Choose a family and a strategy; its description, FSBI input, instrument and key metrics appear first, then three sections underneath:
   1. **Backtest**: growth or cumulative $, drawdown, monthly results, optional comparison strategies.
   2. **Strategy lab**: rebuilds the strategy from its raw signals with your settings. Forecast strategies take forecast source, signal, threshold, window, cost and a scenario shift. Large vs small takes signal, horizon and threshold. FSBI sector strategies take construction, ranking signal (including PCA and pricing-power), leg size, window and cost. Regional breadth takes its thresholds. Single names take basket, signal, horizon, threshold and direction. Strategies whose raw signals are not exported (release gap, Kalshi, overlays) get a return-based lab: direction, sizing, extra cost, vol targeting, loss cap.
   3. **Projection**: a Monte Carlo of the lab version, optionally resampling only months with today's position, which the lab derives from the latest forecast or signal.

## Run it

1. Run the notebook through the cell **"Export for the dashboard"** (end of section 7). It writes `data/dashboard/`.
2. `pip install -r dashboard/requirements.txt`
3. From the project folder: `streamlit run dashboard/app.py`

The app looks for `../data/dashboard` relative to itself. Set `FSBI_DASH_DATA` or paste a path into the sidebar if your data folder is elsewhere. `FSBI_DASH_DATA=data/dashboard python dashboard/test_app.py` walks every page and every strategy and checks that each chart contains data.
