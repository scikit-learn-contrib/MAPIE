# AGENTS.md

This file provides guidance to coding agents working in this repository. For the
human-oriented contribution workflow (forking, PRs, validation process), see
[CONTRIBUTING.md](CONTRIBUTING.md).

## What MAPIE is

MAPIE is a scikit-learn-compatible library for uncertainty quantification (conformal prediction) and risk control. It wraps any base estimator to produce prediction intervals (regression), prediction sets (classification), or risk-controlled decisions, with distribution-free theoretical guarantees. The library is at v1, which introduced a new public API distinct from the legacy one (see "v1 API" below).

## Working principles

- Prefer clean, minimal, readable code. The simplest solution that fully solves the problem is the best one.
- Make the smallest change that accomplishes the task. Don't refactor, rename, or reformat code unrelated to what you were asked to do.
- Match the style, naming, and idioms of the surrounding code rather than introducing new patterns.

## Environment & common commands

The project uses `uv`. The venv lives in `.venv`. Either activate it (`source .venv/bin/activate`) or prefix commands with `uv run`.

**Do not modify the venv** (installing/upgrading/removing packages, re-running `uv sync`, etc.) without asking first.

Initial setup: `uv sync --python 3.13 --all-extras`

All routine checks go through the `Makefile` (these are exactly what CI runs):

- `make lint` — `ruff check` on `examples mapie notebooks`
- `make format` — `ruff format --check` (use `make format-fix` to auto-format before committing)
- `make type-check` — `mypy mapie`
- `make coverage` — full test suite with **100% coverage required** (`--cov-fail-under=100`); also runs `--doctest-modules`. This is the final check before considering a change done — it runs the whole suite, so there's no need to also run `make tests`.
- `make tests` — same suite without the coverage gate; faster, useful only while iterating.
- `make all-checks` — lint + type-check + coverage

The full suite can be slow. While iterating — especially on a minor change — run only the relevant tests (see below) and save `make coverage` for a final pass before you're done.

Run a single test (the suite is invoked via `--pyargs mapie`, not a path):
```sh
pytest mapie/tests/test_regression.py
pytest mapie/tests/test_regression.py -k "test_name_substring"
```

Doctests are part of CI (`--doctest-modules`); docstring `>>>` examples in public classes are executed and must pass.

Long-running tests live in `mapie/tests/long_tests` and are excluded from the default runs — `make long-tests` runs them.

Pre-commit hooks (format-fix, lint, type-check; not tests) can be installed with `uv run pre-commit install`.

## Conventions enforced by CI

- Coverage must be exactly 100%. New code without covering tests will fail CI.
- `ruff` lint rule set is narrow (`E4`, `E7`, `E9`, `F`); line length 88; target Python 3.9 (the library must run on 3.9+).
- Public classes/functions need numpy-style docstrings with a runnable doctest. `BinaryClassificationController` is cited as the reference example.
- Any estimator must follow the scikit-learn API (`BaseEstimator` + the relevant mixin, `clone`-able, `check_is_fitted`).
- User-facing changes get a line in `HISTORY.md`.
- A new method should be backed by a peer-reviewed publication; otherwise it goes under `mapie/experimental/`.

## Architecture

The codebase has two layers: thin **public API classes** that own the user-facing workflow, and internal estimator/conformity-score machinery they delegate to.

### v1 public API — the fit → conformalize → predict workflow

The v1 classes split the lifecycle into explicit steps rather than a single `fit`:
1. `fit(X_train, y_train)` — fit the base estimator (skipped when `prefit=True`).
2. `conformalize(X_conf, y_conf)` — compute conformity scores on a held-out conformalization set.
3. `predict_interval` / `predict_set` — return points plus intervals/sets.

Use `train_conformalize_test_split` (in `mapie/utils.py`) to produce the three splits.

Public classes:
- Regression (`mapie/regression/regression.py`): `SplitConformalRegressor`, `CrossConformalRegressor`, `JackknifeAfterBootstrapRegressor`.
- Quantile regression (`mapie/regression/quantile_regression.py`): `ConformalizedQuantileRegressor`.
- Classification (`mapie/classification.py`): `SplitConformalClassifier`, `CrossConformalClassifier`.
- Calibration (`mapie/calibration.py`): `TopLabelCalibrator`, `VennAbersCalibrator`.

These public classes are wrappers. The actual estimation lives in internal classes prefixed with `_Mapie` (`_MapieRegressor`, `_MapieClassifier`, `_MapieQuantileRegressor`). The legacy v0 API was built directly on the `_Mapie*` classes; `TimeSeriesRegressor` (`mapie/regression/time_series_regression.py`) still subclasses `_MapieRegressor` directly. When changing behavior, check whether logic belongs in the public wrapper or the shared `_Mapie*` core.

### Conformity scores (`mapie/conformity_scores/`)

This is the extension point for new methods. Hierarchy:
- `BaseConformityScore` (`interface.py`) → `BaseRegressionScore` (`regression.py`) and `BaseClassificationScore` (`classification.py`).
- Regression scores in `bounds/`: `AbsoluteConformityScore`, `GammaConformityScore`, `ResidualNormalisedScore`. A new regression score implements `get_signed_conformity_scores` and `get_estimation_distribution`.
- Classification scores in `sets/`: `LACConformityScore`, `NaiveConformityScore`, `APSConformityScore` (extends Naive), `RAPSConformityScore` (extends APS), `TopKConformityScore`. These implement `get_predictions`, `get_conformity_score_quantiles`, `get_prediction_sets`.

Public API accepts these either as a string alias (e.g. `"absolute"`, `"gamma"`) or an instance; resolution happens in `conformity_scores/utils.py`.

### Estimators (`mapie/estimator/`)

`EnsembleRegressor` and the classifier estimator encapsulate the cross-validation / bootstrap fitting strategy (single split, CV, jackknife-after-bootstrap) and produce the out-of-fold predictions the conformity scores consume. `mapie/subsample.py` provides the `Subsample` resampler for jackknife-after-bootstrap.

### Risk control (`mapie/risk_control/`)

Separate from interval/set prediction. Controller classes — `BinaryClassificationController`, `MultiLabelClassificationController`, `SemanticSegmentationController` — calibrate a threshold/parameter to control a risk (`risks.py`) with statistical guarantees, using FWER procedures in `fwer_control.py`.

### Other modules

- `mapie/exchangeability_testing/` — tests for whether the exchangeability assumption (required for valid conformal prediction) holds.
- `mapie/experimental/` — unvalidated methods (e.g. PyTorch-based standardized residuals); depends on the `experimental`/`dev` torch extra.
- `mapie/metrics/` — coverage and interval-width metrics.

## Documentation

Two doc systems coexist: `doc/` is the current MkDocs site (`mkdocs build --strict`, `mkdocs serve`); `doc_legacy/` is the older Sphinx site (`make doc-legacy`, `make doctest`). Runnable examples in `examples/` are auto-included in the docs.
