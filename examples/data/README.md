# Example datasets

This folder stores datasets used by the examples in `examples/`. They are
loaded by default so that the examples run without depending on the
availability of external download servers.

## `demand_temperature.csv`

Hourly electricity demand (GW) and temperature (°C) for the state of Victoria,
Australia. Used by the time-series examples
(`examples/regression/1-quickstart/plot_ts-tutorial.py` and
`examples/regression/2-advanced-analysis/plot_timeseries_enbpi.py`).

## `blogData_train.zip`

Backup of the **BlogFeedback** training set, used by
`examples/regression/3-scientific-articles/plot_kim2020_simulations.py`.

- Zip archive containing a single CSV, `blogData_train.csv`.
- Shape: 52,397 rows × 281 columns (280 features + 1 target), no header.
- The target (last column) is the number of comments a blog post received in
  the next 24 hours; the example models `log(1 + target)`.
- Read with `pandas.read_csv(ZipFile(...).open("blogData_train.csv"),
  header=None)`.

The original dataset comes from the UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/304/blogfeedback>
(Buza, K. (2014). BlogFeedback [Dataset]. DOI: 10.24432/C58S3F).

The example loads this backup by default. To fetch the original archive from
UCI instead, call `get_X_y(download=True)`.
