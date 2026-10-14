# Research and Trading
## Asset Classes
- Ultrashort Treasury Inflation-Protected Security (TIPS) ETF (RBIL)
    - **RBIL tracks the Bloomberg U.S. Ultrashort TIPS 1–13 Months Index.**
    - A passively managed exchange-traded fund that tracks the Bloomberg U.S. Ultrashort TIPS 1-13 Months Total Return Unhedged USD Index. It holds U.S. Treasury Inflation-Protected Securities maturing in 13 months or less, with an average duration well under one year, so it is much less sensitive to interest rate movements than longer-dated TIPS. The fund is designed to protect purchasing power through CPI-indexed principal while minimizing duration risk, making it similar in character to an inflation-indexed Treasury bill. It has a total expense ratio of 0.17% and pays monthly distributions.

- iShares 1-3 Year Treasury Bond ETF (SHY)
    - **SHY tracks the ICE U.S. Treasury 1–3 Year Bond Index.**
    - The ICE U.S. Treasury 1-3 Year Bond Index consists of U.S. Treasury bonds with remaining maturities between one and three years. It provides targeted exposure to short-term U.S. government debt and has an effective duration of approximately 1.81 years. With a very low expense ratio of 0.15%, SHY offers exposure to short-term Treasuries, which generally carry lower credit and interest rate risk than longer-dated bonds.

- iShares 0-5 Year TIPS Bond ETF (STIP)
    - **STIP tracks an index of short-maturity TIPS.**
    - An index of inflation-protected U.S. Treasury bonds with remaining maturities of five years or less provides a way to hedge against inflation. This hedge manmanages duration risk by focusing on shorter-maturity TIPS, whose principal adjusts with inflation. This shorter-dated focus helps reduce the interest rate risk that longer-dated TIPS can introduce.

- Kalshi Binary Options
    - Kalshi binary options are CFTC-regulated event contracts that operate as all-or-nothing positions on real-world events. A trader purchases a "Yes" or "No" contract at a price between $0.01 and $0.99, and the contract settles at $1 if correct or $0 if incorrect. Kalshi runs these contracts through its own regulated exchange infrastructure, with dollar-denominated settlements and a one-dollar notional value per contract, allowing participants to trade on outcomes ranging from economic data releases to election results and sports championships.
    
- State Street Consumer Staples Select Sector SPDR ETF (XLP)
    - **XLP tracks the Consumer Staples Select Sector Index.**
    - The Consumer Staples Select Sector Index represents the consumer staples sector of the S&P 500 Index. It provides exposure to companies in consumer staples distribution and retail, household products, food products, beverages, tobacco, and personal care products. As a defensive sector fund, XLP is often used for its lower volatility compared with more cyclical sectors.

- State Street Consumer Discretionary Select Sector SPDR ETF (XLY)
    - **XLY tracks the Consumer Discretionary Select Sector Index.**
    - The Consumer Discretionary Select Sector Index represents the consumer discretionary sector of the S&P 500 Index. It provides exposure to companies in specialty retail, broadline retail, hotels, restaurants and leisure, textiles, apparel and luxury goods, household durables, automobiles, and related industries. As a more cyclical sector fund, XLY tends to be more sensitive to economic growth and inflation dynamics than defensive sector funds like XLP.


## Signal & Value Definitions

**Definitions:**
- $CPI_{Pred}$
    - % change in CPI, month-over-month, as predicted by FSBI, macro, and other data in an ML model.
- $CPI_{Real}$
    - Actual % change in CPI, month-over-month.
- $CPI_{Kalshi}$
    - % change in CPI, month-over-month, as inferred by Kalshi prices
- $CPI_{Nowcast}$
    - % change in CPI, month-over-month, provided by the Fed.

**Residuals**
- 
$$
\begin{align*}
\text{Prediction Residual: } &e_{Pred} &&= CPI_{Pred} - CPI_{Real} \\
\text{Kalshi Residual: } &e_{Kalshi} &&= CPI_{Kalshi} - CPI_{Real} \\
\text{Nowcast Residual: } &e_{Nowcast} &&= CPI_{Nowcast} - CPI_{Real} \\
\end{align*}
$$ 

**Key Differences**
$$
\begin{align*}
\text{Market Disagreement: } & MD &&= CPI_{Pred} - CPI_{Real} \\
\text{Nowcast Disagreement: } & ND &&= CPI_{Pred} - CPI_{Nowcast} \\
\text{Market-Fed Disagreement: } & MFD &&= CPI_{Pred} - CPI_{Nowcast} \\
\end{align*}
$$ 


## Strategies

### 1. Kalshi Directional
1. **Background**

2. **Instrument(s)**

3. **Signal**
4. **Associated Literature**

    [1] Kalshi Research, “Beyond consensus: Prediction markets and the forecasting of inflation shocks,” Kalshi Research, Jan. 2025. [Online]. Available: https://kalshi.com/research/publications/crisis-alpha

    - Kalshi Research (2025) compares Kalshi prediction-market implied CPI forecasts with institutional consensus forecasts for year-over-year CPI from February 2023 to mid-2025. It finds that Kalshi forecasts have 40.1% lower mean absolute error overall and outperform consensus especially during inflation shocks: roughly 50% lower MAE for shocks above 0.2 pp and for 0.1–0.2 pp shocks one week ahead, rising to about 60% and 56.2% lower one day before release. When markets and consensus disagree, market predictions are closer in 75% of cases, and market-consensus deviations above 0.1 pp predict an approximately 81.2% shock rate, increasing to 82.4% one day prior. The paper attributes this “shock alpha” to crowd wisdom, heterogeneous information, stronger accuracy incentives, and better aggregation, concluding that prediction markets are a valuable complement to consensus forecasts, particularly in tail events, though the sample is limited.

    [2] Underlying Inflation, “Proprietary US inflation forecasts,” Proprietary Investor Research, White Paper, 2026.

    - Underlying Inflation (2026) compares proprietary bottom-up and machine-learning US inflation models against Consensus and Kalshi market-implied CPI forecasts using real-time data from January 2023 onward. Ten days before release, the bottom-up model has the lowest pooled MAE of 0.070, ahead of Consensus at 0.077, ML at 0.082, and Kalshi at 0.097; the ML model is strongest for headline year-over-year CPI with an MAE of 0.084, beating Consensus at 0.090 and Kalshi at 0.092. Kalshi improves as the release approaches, with pooled MAE falling to 0.089 at one day and one hour, but it generally remains behind the proprietary models, especially for YoY inflation. The models also provide directional value when Kalshi is surprised: bottom-up is correct 72.4% of the time and 12 out of 12 times when the model-Kalshi gap is at least 0.2 pp. Backtests show profitable trading strategies, with a 50/50 ensemble and ML-based distributional signals performing well, and Kalshi prices tending to converge toward the proprietary forecasts before release. The paper cautions that these are historical simulations with a short sample, not a live track record.


### 2. Ultrashort TIPS (Long Only)
1. **Background (Shared with Strategy 3)**

    When held until maturity, TIPS provide a full protection against inflation, delivering the “real yield” as indicated at the time of purchase. On the other hand, a fixed-rate Treasury bond (“Nominal Treasury”) pays a fixed principal and coupon. 

    Shorter-dated TIPS have a stronger correlation with inflation (62%) compared to their longer-dated counterparts and are more responsive to immediate changes in the CPI. The correlation between TIPS and nominal Treasury bonds also varies based on the maturity, with shorter-dated TIPS showing lower correlations to Treasury compared to longer ones.

    In essence, **for strategy 2**, when the CPI increase is notably higher than expected inflation (breakeven), then it is best to go long in RBIL. Otherwise, remain flat.

    The TIPS breakeven is the difference between the nominal Treasury yield and the TIPS real yield at the same maturity. It represents the market's priced-in expectation of average inflation over the bond's life.

    Thus, **in strategy 3**, if the model predicts that CPI will exceed the current breakeven, go long TIPS / short nominal Treasuries (a "long breakeven" position). If the model predicts CPI will fall short, do the reverse.

2. **Instrument**

    Ultrashort Treasury Inflation-Protected Security (TIPS) ETF (RBIL)

3. **Signal**

    $\text{Signal} = CPI_{predicted} − Breakeven$

    $\text{Signal} > \text{Threshold}$ $\implies$ Long RBIL

    Else, exit RBIL position. 
    

### 3. TIPS Breakeven (Long/Short)
1. **Background**

    Reference strategy 2.

    This strategy switches to STIP from RBIL. While RBIL is more responsive to inflation (and therefore CPI), its lower duration causes some trouble. SHY and STIP are relatively comparable in terms of duration, while RBIL is significantly lower. If there is a change in the interest rates, RBIL will not move in a way that "cancels" the change in SHY's value. Meanwhile, STIP, if held at the proper weight, would perfectly offset this change in SHY value (since one asset is shorted while the other is held long). 

    Moreover, RBIL has higher fees and is less liquid, driving a higher bid-ask spread. Both of these would eat away at the alpha.

2. **Instrument(s)**

    TIPS ETF (STIP) and Nominal Treasury ETF (SHY)


3. **Signal**
    
    $\text{Signal} = CPI_{predicted} − Breakeven$
    
    $\text{Signal} > \text{Threshold}$ $\implies$ Long STIP, short SHY

    $\text{Signal} > \text{Threshold}$ $\implies$ Long SHY, short STIP


4. **Associated Literature**

    [1] S. P. Kothari and J. Shanken, “Asset allocation with inflation-protected bonds,” Financial Analysts Journal, vol. 60, no. 1, pp. 54–70, 2004.

    - Kothari and Shanken (2004) examine how the availability of inflation-indexed bonds affects investors’ asset allocation decisions. Since U.S. TIPS were only introduced in 1997, the authors construct a hypothetical series of five-year zero-coupon indexed bond returns back to 1953 by forecasting inflation and backing out real interest rates from nominal Treasury yields, and they also analyze actual TIPS data from February 1997 through July 2003. They find that indexed bonds have lower real return volatility than conventional bonds and that their correlation with stock returns is close to zero, whereas conventional bond returns are significantly positively correlated with stocks. This low correlation means indexed bonds provide better diversification benefits: an equal-weighted stock-bond portfolio has about 13% lower standard deviation when indexed bonds replace conventional bonds, and the risk reduction is even larger for bond-heavy portfolios. Mean-variance analysis suggests substantial allocations to indexed bonds—often 40–70% of the portfolio—depending on assumptions about the inflation risk premium, while conventional bonds receive little or no weight unless a large positive inflation risk premium is assumed. The authors conclude that indexed bonds should play a significant role in efficient portfolios, though they caution that their analysis is preliminary and based on a one-year horizon with only stocks and bonds.

    [2] A. Andonov, F. Bardong, and T. Lehnert, “TIPS, inflation expectations, and the financial crisis,” Financial Analysts Journal, vol. 66, no. 6, pp. 27–39, 2010.

    - Andonov, Bardong, and Lehnert (2010) investigate whether the U.S. TIPS market is efficient and whether inefficiencies can be exploited by informed traders, with particular attention to the 2008 financial crisis. They compare breakeven inflation (the difference between nominal and TIPS yields) with inflation forecasts from the Survey of Professional Forecasters and an adjusted Kothari-Shanken inflation model, then test long-short TIPS strategies and breakeven trading strategies over 3-, 6-, and 12-month holding periods from January 1997 to June 2009. They find persistent positive excess returns, especially for breakeven trades using Survey forecasts, which earned up to about 3.5% per year even after transaction costs, providing evidence against semi-strong market efficiency. During 2008, breakeven inflation turned negative for some maturities—implying expected deflation—which the authors attribute to a sharp increase in the liquidity premium rather than genuine deflation expectations, since TIPS are less liquid than nominal Treasuries and investors fled to liquidity during the crisis. The paper concludes that the TIPS market has not yet fulfilled its promise as a low-risk, efficient, and liquid instrument, and that systematic inflation forecasting is particularly valuable during periods of macroeconomic uncertainty.

    [3] S. Atha and V. Jain, “Tips for TIPS: Ultrashort maturities for inflation,” Bloomberg Professional Services, Mar. 2025. [Online]. Available: https://www.bloomberg.com/professional/insights/

    - Atha and Jain (2025) explain the important distinction between shorter-dated and longer-dated TIPS for inflation hedging, emphasizing that duration exposure in longer-dated TIPS can introduce interest rate risk that offsets inflation protection. They note that TIPS have longer duration than similar-maturity nominal bonds because coupons are lower and principal is adjusted upward for inflation, and that this duration risk became evident in 2022 when the Bloomberg TIPS Index fell 11.8% despite CPI rising 6.5% year-over-year. Using Bloomberg indices, they show that ultrashort TIPS (maturing in 1–13 months) have a much higher correlation with inflation (62%) than longer-dated TIPS, and a lower correlation with nominal Treasuries, making them more responsive to immediate CPI changes. Their analysis of inflation shocks—defined as the difference between one-year breakeven inflation swaps and subsequent realized CPI—shows that ultrashort TIPS tend to generate higher returns when CPI exceeds breakeven inflation. The authors conclude that investors seeking pure inflation protection with minimal interest rate risk should consider shorter-dated TIPS, while those seeking to hedge long-term inflation expectations or real returns should consider longer-dated TIPS or breakeven inflation swaps.


### 4. XLP/XLY Sector Rotation
1. **Background**

    When people expect the value of a dollar bill to remain relatively stable, they are more willing to spend money on frivolous goods, and when they are not confident that inflation will be controlled, they pull-back on personal expenses dramatically. 

2. **Instrument(s)**
    
    XLP (Consumer Defensive) and XLY (Consumer Discretionary)

    Approach A: When CPI fluctuates above a certain threshold in either direction, XLP is a safer bet. When CPI is expected to remain relatively stable, XLY is expected to perform better.

    Approach B: When CPI is surprisingly high (prediction is greater than Kalshi estimate), then it means that inflation was higher than expected. The literature suggests that discretionary spending is expected to be more sensitive to high inflation that staples. Rotate to XLY when CPI is expected to be surprisingly low.

3. **Signal**

    This rotation occurs shortly before the official CPI report is released and held throughout the month.
    
    Approach A:

    $\vert CPI_{Pred} \vert > Threshold \implies$  Long XLP

    $\vert CPI_{Pred} \vert > Threshold \implies$  Long XLY

    Approach B:

    $CPI_{Pred} - CPI_{Kalshi} > 0 \implies$ Long XLP

    $CPI_{Pred} - CPI_{Kalshi} < 0 \implies$ Long XLY

4. **Associated Literature**

    [1] A. Gupta, A. Kassam, R. Suryanarayanan, and K. Varga, “Index performance in changing economic environments: A macroeconomic framework,” MSCI Research Insight, MSCI Inc., Apr. 2014.

    - Gupta et al. (2014) develop a macroeconomic framework for analyzing how MSCI equity factor and sector indexes perform under changing economic growth and inflation regimes. Using over 40 years of historical MSCI World index data and long-term forecasts from MSCI’s Macroeconomic and Asset Pricing Models, the paper classifies indexes as cyclical or defensive and as growth-sensitive or growth-hedging. Historically, Industrials, Consumer Discretionary, and Information Technology are the most cyclical sectors, while Health Care, Consumer Staples, and Utilities are the most defensive; among factors, Equal Weighted and Value Weighted are most cyclical, and Minimum Volatility and Quality are most defensive. Using a bivariate CLI/CPI regime framework, the paper finds that Momentum and Equal Weighted tend to perform best in “Heating Up” and “Goldilocks” regimes, Minimum Volatility performs best in “Stagflation,” and defensive sectors outperform in “Slow Growth.” Regime-transition results show that Equal Weighted tends to outperform when growth remains positive or strengthens, Quality when growth slows, Minimum Volatility when slow growth is coupled with low inflation, and Energy during stagflation. The paper concludes that macroeconomic views and tolerance for macroeconomic uncertainty should inform tilts between growth-sensitive and growth-hedging factor and sector indexes.

    [2] A. Valadkhani, “Inflation-driven instability in US sectoral betas,” Journal of Asset Management, vol. 26, no. 5, pp. 509–516, 2025.

    - Valadkhani (2025) examines how inflation affects sectoral betas for nine US SPDR sector ETFs and the S&P 500 from January 1999 to February 2024 using a functional coefficient model in which beta is allowed to vary with annualized inflation. The study finds that sectoral betas remain stable and meaningful only when inflation stays within roughly a 1–5% range. Defensive sectors such as Consumer Staples, Utilities, and Health Care have lower betas and tolerate a wider inflation range, while more inflation-sensitive sectors such as Consumer Discretionary, Technology, Financials, and Energy have higher betas and narrower stable ranges. Once inflation moves outside this range—especially above 4–5%—betas become erratic and unreliable across all sectors, undermining beta as a measure of systemic risk and making sector-based diversification difficult. The paper concludes that the Federal Reserve’s 2% inflation target is important for preserving the reliability of risk measures and maintaining financial market stability.

### 5. Bitcoin
1. **Background**

    **Bitcoin is not presently viable for a trading strategy.**

    Bitcoin has only shown to be a hedge against inflation during COVID-19, but after that regime, the relationship has disappeared. This can be credited to institutional adoption of BTC and changes in the perspectives of retail investors.

    Other research shows that BTC falls in response to positive inflation surprises. The relationship is, nonetheless, inconsistent and likely regime dependent. Team 3 showed disappointing results with this asset class. Most research shows a statistically insignificant relationship between CPI and BTC. Most correlations are the results of random noise and confounding variables (like unemployment).

    There may be some merit to using $CPI_{Kalshi}$ with out estimate for altcoin price volatility, but this could be quite the rabbit hole. It is best to make something work before we explore this.
    

2. **Instrument(s)**

3. **Signal**

4. **Associated Literature**

    [1] M. Pinchuk, “Bitcoin does not hedge inflation,” Working Paper, May 2021. [Online]. Available: https://ssrn.com/abstract=3862182. doi: 10.2139/ssrn.3862182.

    - Pinchuk (2021) studies how major cryptocurrencies react to macroeconomic news announcements and finds that Bitcoin is the only one with a significant response: it falls by about 24 basis points after a one-standard-deviation inflation surprise. This contradicts the popular view that Bitcoin is an inflation hedge. The paper does not find support for an interest-rate explanation, but instead suggests that higher inflation reduces savings and therefore lowers inflows into Bitcoin. Overall, it concludes that investors should not rely on Bitcoin to hedge inflation or as a stable store of value.

    [2] H. Rodriguez and J. Colombo, “Is bitcoin an inflation hedge?,” Journal of Economics and Business, vol. 133, Art. no. 106218, 2025, doi: 10.1016/j.jeconbus.2024.106218.

    - Rodriguez and Colombo (2025) use monthly VAR models from 2010 to 2023 to test whether Bitcoin hedges unexpected inflation, measured as surprises in CPI and Core PCE announcements. They find that Bitcoin returns tend to rise after positive CPI inflation surprises, behaving somewhat like gold, which suggests Bitcoin can act as an inflation hedge. However, this result depends on the inflation index used and the sample period: it does not hold for Core PCE surprises and largely disappears after the COVID-19 outbreak and Bitcoin’s broader institutional adoption. They conclude that Bitcoin’s inflation-hedging property is context-specific and less reliable than gold’s.

### 6. Volatility Trade
1. **Background**

    **Market volatility is not presently viable for a trading strategy.**

    When CPI is announced, there is almost always a decerase in volatility. This is due to investors hedging against a black swan event prior to metric release. The sudden drop in demand for insurance (options) against adverse moves causes option prices to fall, dragging down implied volatility. Even when CPI spikes above investor expectations, the volatility crush post-event is so powerful that VIX and VXX fall. 

    It is like preparing to die but only breaking your arm. Sure, you broke your arm, but what a relief that you're not dead!

    Only when CPI spikes the degree of a "tail event" that volatility increases meaninfully. We do not have enough data to support a trading strategy based around tail events, especially when the norm of the strategy is tantamount to "short until we think that's a bad idea." It's a great way to become insolvent.

    
    "Ederington and Lee (1996) find that, on average, implied volatility in financial markets decrease on days with announcements regardless of the content of the announcement."

    "Effectively, the surprise factor is negative if any realized figure is lower than market expectations."

    ""

2. **Instrument(s)**

3. **Signal**

4. **Associated Literature**

    [1] L. H. Ederington and J. H. Lee, “The creation and resolution of market uncertainty: The impact of information releases on implied volatility,” Journal of Financial and Quantitative Analysis, vol. 31, no. 4, pp. 513–539, Dec. 1996.

    - Ederington and Lee (1996) examine how scheduled and unscheduled information releases affect market uncertainty, measured by the implied standard deviation (ISD) from options on T-Bond, Eurodollar, and Deutschmark futures from November 1988 to September 1992. They hypothesize that because the timing of scheduled announcements is known in advance, pre-release implied volatility already reflects the expected high-volatility event; once the announcement occurs and uncertainty is resolved, the ISD tends to fall. This is the original “volatility crush” result. In contrast, unscheduled announcements are surprises, so the unexpectedly high actual volatility leads market participants to revise upward their expectations of future volatility, causing the ISD to rise. Empirically, they find that ISDs fall on scheduled announcement days—especially for the employment report, PPI, and CPI—and rise on non-announcement days, with larger declines for shorter-dated options and announcements that normally move prices more. They also show that the well-known Friday-decline/Monday-rise pattern in implied volatility is largely due to the bunching of scheduled announcements on Fridays, not buying/selling pressure. Finally, they find that the predictable ISD decline after employment reports cannot be profitably exploited because transaction costs and, more importantly, convexity losses from large price swings offset the gains.


    [2] L. Tuysuzoglu, “The effects of macroeconomic data surprises on implied volatility,” B.S. honors thesis, Leonard N. Stern School of Business, New York University, New York, NY, USA, May 2008.

    - Tuysuzoglu (2008) tests whether macroeconomic data surprises move implied volatility (VIX) and the S&P 500 in the expected direction: negative surprises should raise VIX and lower the S&P 500, while positive surprises should do the opposite. Using nine U.S. macro releases—CPI, core CPI, non-farm payrolls, housing starts, GDP, retail sales, Leading Indicators, capacity utilization, and construction spending—from January 2003 to December 2007, and measuring surprises against both autoregressive forecasts and published expectations, the paper finds that only a few announcements matter. Positive non-farm payroll surprises significantly reduce VIX, meaning weak payroll news raises implied volatility, and core CPI surprises also affect VIX; core CPI surprises are the only macro variable that significantly moves the S&P 500. Most other macroeconomic surprises have no statistically significant effect on either VIX or the S&P 500. The results support prior evidence that employment and inflation news are the most important macro releases for option-implied volatility, while most scheduled data releases do not generate predictable changes in VIX.

    