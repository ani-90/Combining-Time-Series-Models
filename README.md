# Combining Time Series Models

**Forecasting index-level closing prices by clustering correlated constituents and combining their ARIMA forecasts.**

Undergraduate research project (PSG College of Technology). Studies whether a weighted combination of per-stock ARIMA forecasts predicts a constituent's closing price better than that stock's own model does — using the BSE BANKEX and BSE Auto indices as test beds.

---

## The question

A stock index is not a bag of independent series. Bank stocks move together; auto stocks move together. If those correlations carry information, a forecast for HDFC should be improvable by looking at what the models for Axis, ICICI, Kotak, AU, and SBI are simultaneously predicting.

This project makes that concrete in four steps: measure the correlation structure, use it to reduce redundancy, fit a univariate model per remaining series, then search for the weighted blend of those forecasts that best reproduces each target.

---

## Method

**1. Correlation structure**

Daily closing prices are loaded per company and assembled into a single frame. From the Pearson correlation matrix, a distance matrix is derived:

```
d(i,j) = 1 − r(i,j)²
```

Squaring removes sign, so two stocks that move in opposite directions with equal consistency are treated as equally *informative* about each other. Pairs are then bucketed — strong (d ≤ 0.36), moderate (0.36 < d < 0.75), weak (d ≥ 0.75).

**2. Structure discovery**

Three views of the same distance matrix:

- **PCA** — how many components explain the variance across the sector, i.e. how many genuinely independent factors drive it
- **Hierarchical clustering** — dendrograms showing which companies group and at what distance
- **Minimum spanning tree** — the sector's correlation backbone as a graph, built with `scipy.sparse.csgraph.minimum_spanning_tree` and drawn with `networkx`

**3. Series reduction**

Pairs identified as strongly related are merged into one series by taking the elementwise maximum of the two closing prices — for BANKEX, (Axis, BoB), (Federal, ICICI), and (IDFC, IndusInd), collapsing 10 series into 7. This keeps the sector's dominant movement while dropping near-duplicate columns before modelling.

**4. Univariate modelling**

Each series is decomposed (`seasonal_decompose`, additive), inspected via ACF/PACF, and first-differenced to reach stationarity. An ARMA order is then selected per series by grid search over p, q ∈ {0, 1, 2}, choosing the lowest AIC. `pmdarima.auto_arima` is used as a cross-check on the selected orders.

**5. Forecast combination**

Each fitted model forecasts the 20 trading days of December 2023 — data held out entirely from fitting. For a target series, the combined prediction is

```
ŷ_target(t) = Σ_i w_i · ŷ_i(t)
```

Two weighting schemes are evaluated:

- **Likelihood weighting** — each model's weight is `σ_i^(-1/2) · exp(−½ Σ (y − ŷ_i)² / σ_i²)`, the Gaussian likelihood of the observed December values under that model's forecast variance. Models whose forecasts were both confident and correct dominate.
- **Constrained random search** — 100 weight vectors sampled on the simplex, keeping the vector with lowest MSE against the target, subject to the target's own model carrying the largest weight. The constraint prevents the blend from degenerating into "predict HDFC using only SBI."

---

## Repository contents

| File | What it does |
|---|---|
| `bankex.ipynb` | Correlation, distance matrix, PCA on the 10 BANKEX banks |
| `spanning tree for bankex.ipynb` | MST + graph view of the banking sector |
| `auto_index dendrogram.ipynb` | Hierarchical clustering of the 15 auto constituents |
| `auto_index spanning_tree.ipynb` | MST for the auto sector |
| `time series with bankex data.ipynb` | Full pipeline — differencing, ARIMA selection, December forecasts, weight search |
| `time series with auto index data.ipynb` | Same pipeline applied to the auto index |
| `*_research_paper_*.doc(x)` | IEEE-format write-up |

**Reading order:** `bankex.ipynb` → `spanning tree for bankex.ipynb` → `time series with bankex data.ipynb`. The auto-index notebooks repeat the method on a second sector.

---

## Data

Daily closing prices, January 2022 – November 2023, sourced from BSE historical data:

- **BANKEX (10):** AU, Axis, Bank of Baroda, Federal, HDFC, ICICI, IDFC First, IndusInd, Kotak, SBI
- **Auto (15):** Apollo Tyres, Ashok Leyland, Bajaj, Balkrishna, Bosch, Cummins, Eicher, Hero, M&M, Maruti, Motherson, MRF, Tata Motors, Tube Investments, TVS

December 2023 closing prices are held out as the evaluation set.

> **The CSVs are not committed to this repository.** The notebooks were written in Google Colab and read from `/content/<company>.csv`, each with a `Close Price` column. To reproduce, download the equivalent BSE data and place it where the notebooks expect it, or edit the paths to a local `data/` directory.

**Environment:**

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy networkx pmdarima
```

---

## Findings

The correlation structure inside both indices is strong enough that several constituents carry near-duplicate information — the PCA and MST results both show a small number of factors and a tight backbone rather than a diffuse graph. Blending correlated constituents' ARIMA forecasts reduces error against a single-series ARIMA on the December hold-out, with the improvement concentrated in stocks that sit centrally in the MST.

*Numbers, plots, and the full discussion are in the research paper included in this repo.*

---

## Known limitations

- **Short evaluation window.** 20 trading days of a single month is too small to separate genuine improvement from luck. A rolling-origin evaluation across multiple months is the honest test.
- **Weights are fitted on the evaluation period.** Both schemes select weights using December actuals and are then assessed on December — so the reported improvement is in-sample for the combination step and optimistic. A separate validation split for weight fitting is needed.
- **Random search is shallow.** 100 samples over a 6-simplex; convex optimisation would find the true optimum directly.
- **Elementwise max as a merge rule** is a heuristic without a strong statistical justification. Averaging, or a PCA factor over the pair, would be more defensible.
- **`freq='D'` date ranges** are assigned to trading-day data, so the date index drifts from actual market days. This affects labelling, not the fitted values.
- **No transaction costs, no out-of-sample trading test.** This forecasts prices; it is not a strategy and makes no claim about profitability.

---


## Citation

Anirudh Balakrishnan *Combining Time Series Models to Compute Closing Stock Price.* PSG College of Technology.

---

## Licence

MIT
