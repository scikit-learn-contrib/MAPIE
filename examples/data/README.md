# Example datasets

This folder stores datasets used by the examples in `examples/`. They are
loaded by default so that the examples run without depending on the
availability of external download servers.

## `demand_temperature.csv`

Hourly electricity demand (GW) and temperature (°C) for the state of Victoria,
Australia. Used by the time-series examples
(`examples/regression/1-quickstart/plot_ts-tutorial.py` and
`examples/regression/2-advanced-analysis/plot_timeseries_enbpi.py`).

## `blogData_train.csv.gz`

Backup of the **BlogFeedback** training set, used by
`examples/regression/3-scientific-articles/plot_kim2020_simulations.py`.

- Gzip-compressed CSV; read directly with `pandas.read_csv(..., header=None)`
  (compression is inferred from the `.gz` extension).
- Shape: 52,397 rows × 281 columns (280 features + 1 target), no header.
- The target (last column) is the number of comments a blog post received in
  the next 24 hours; the example models `log(1 + target)`.

The original dataset comes from the UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/304/blogfeedback>
(Buza, K. (2014). BlogFeedback [Dataset]. DOI: 10.24432/C58S3F).

The example loads this backup by default. To fetch the original archive from
UCI instead, call `get_X_y(download=True)`.

## `zaffran2022_aci_reference.csv`

Backup of the reference adaptive conformal inference (ACI) prediction interval
bounds from Zaffran et al. (2022), used by
`examples/regression/3-scientific-articles/plot_zaffran2022_comparison.py` to
check that MAPIE reproduces the authors' results.

- CSV with two columns, `Y_inf` and `Y_sup` (lower and upper interval bounds).
- One row per prediction step.

The bounds were extracted from the original `ACP_0.04_RF.pkl` pickle hosted in
the authors' repository:
<https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries>
(commit `131656fe4c25251bad745f52db3c2d7cb1c24bbb`,
`results/Spot_France_Hour_0_train_2019-01-01/`).

The example loads this backup by default. To fetch the original pickle from the
authors' repository instead, call `get_reference_results(download=True)`.

## `zaffran2022_prices.csv.gz`

Backup of the French electricity spot-price dataset (with engineered calendar
and lag features) from Zaffran et al. (2022), used by
`examples/regression/3-scientific-articles/plot_zaffran2022_comparison.py`.

- Gzip-compressed CSV; read directly with `pandas.read_csv(...)` (compression
  is inferred from the `.gz` extension).
- Shape: 34,896 rows × 59 columns (hourly data, 2016–2019).
- Columns include `Date`, `Spot` (spot price), `hour`, day-of-week indicators
  (`dow_0`…`dow_6`), 24-hour and 168-hour lag features (`lag_24_*`,
  `lag_168_*`) and consumption (`conso`).

Original file (`data_prices/Prices_2016_2019_extract.csv`) from the authors'
repository:
<https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries>
(commit `131656fe4c25251bad745f52db3c2d7cb1c24bbb`).

The example loads this backup by default. To fetch the original CSV from the
authors' repository instead, call `get_data(download=True)`.
