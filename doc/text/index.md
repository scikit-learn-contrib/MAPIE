---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![MAPIE Logo](../images/mapie_logo_nobg_cut.png){ width="400" }

# MAPIE — Model Agnostic Prediction Interval Estimator

**An open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.**

[![GitHub Actions](https://github.com/scikit-learn-contrib/MAPIE/actions/workflows/test.yml/badge.svg)](https://github.com/scikit-learn-contrib/MAPIE/actions)
[![Codecov](https://codecov.io/gh/scikit-learn-contrib/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1)](https://codecov.io/gh/scikit-learn-contrib/MAPIE)
[![License](https://img.shields.io/github/license/scikit-learn-contrib/MAPIE)](https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mapie)](https://pypi.org/project/mapie/)
[![Python](https://img.shields.io/pypi/pyversions/mapie)](https://pypi.org/project/mapie/)
[![Downloads](https://img.shields.io/pypi/dm/mapie)](https://pypistats.org/packages/mapie)
[![Conda](https://img.shields.io/conda/vn/conda-forge/mapie)](https://anaconda.org/conda-forge/mapie)

[Get Started :material-rocket-launch:](getting-started/quick-start.md){ .md-button .md-button--primary }
[API Reference :material-book-open-variant:](../api/index.md){ .md-button }

</div>

---

<div class="announcement" markdown>
:tada: **MAPIE v1 is live!** This new version introduces major changes to the API. Check out the [release notes](getting-started/v1-release-notes.md).

:rocket: **MAPIE in 2026** — Explore new support for **risk control** in
LLM-as-Judge and image segmentation, **exchangeability tests**, and adaptive
conformal prediction methods. See [GitHub Releases](https://github.com/scikit-learn-contrib/MAPIE/releases)
for the latest changes.
</div>

---

![Educational Visual](../images/educational_visual.png){ width="500", style="display: block; margin: 0 auto;" }

<p style="text-align: center;"><small>Image credits: Cemrecan Yurtman
(portrait) and hogrmahmood (zebra-horse hybrid).</small></p>

## What can MAPIE do?

<div class="grid" markdown>

<div class="card" markdown>

### :material-chart-bell-curve-cumulative: Prediction Intervals & Sets

Compute **prediction intervals** (regression, time series) or **prediction sets** (classification) using state-of-the-art conformal prediction methods.

[Learn more →](conformal-prediction/regression.md)
<br>
[Browse examples →](../generated/regression/index.md)

</div>

<div class="card" markdown>

### :material-shield-check: Risk Control

**Control prediction errors** for complex tasks: multi-label classification, semantic segmentation, with probabilistic guarantees on precision and recall.

[Learn more →](risk-control/theory.md)
<br>
[Browse examples →](../generated/risk_control/index.md)

</div>

<div class="card" markdown>

### :material-puzzle: Model Agnostic

Use **any model** — scikit-learn, TensorFlow, PyTorch — thanks to scikit-learn-compatible wrappers. Part of the **scikit-learn-contrib** ecosystem.

[Get started →](getting-started/quick-start.md)

</div>

<div class="card" markdown>

### :material-school: Theoretically Grounded

Implements **peer-reviewed** algorithms with **theoretical guarantees** under minimal assumptions, based on Conformal Prediction and Distribution-Free Inference.

[Read the theory →](conformal-prediction/regression.md)

</div>

</div>

---

## :art: All Examples

Explore our gallery of hands-on examples covering all MAPIE use cases:

<div class="grid" markdown>

<div class="card" markdown>
### :material-chart-line: Regression
Prediction intervals for regression and time series.

[Browse examples →](../generated/regression/index.md)
</div>

<div class="card" markdown>
### :material-shape: Classification
Prediction sets for single-label and multi-label classification.

[Browse examples →](../generated/classification/index.md)
</div>

<div class="card" markdown>
### :material-shield-alert: Risk Control
Control risks for complex ML tasks with probabilistic guarantees.

[Browse examples →](../generated/risk_control/index.md)
</div>

<div class="card" markdown>
### :material-target: Calibration
Calibrate and evaluate probabilistic predictions.

[Browse examples →](../generated/calibration/index.md)
</div>

<div class="card" markdown>
### :material-swap-horizontal: Exchangeability Testing
Test distribution shifts and monitor exchangeability assumptions.

[Browse examples →](../generated/exchangeability_testing/index.md)
</div>

</div>

---

## :zap: Quick Install

```bash
pip install mapie
```

See the [Quick Start](getting-started/quick-start.md) for other installation
methods, requirements, and a first example.

---

## :memo: Cite MAPIE

Using MAPIE in research? See [Citation and references](about/citation.md)
for the preferred citation, BibTeX, foundational references, and project
affiliations.

---

<div style="text-align: center; color: var(--md-default-fg-color--light); font-size: 0.85rem; margin-top: 2rem;">
  Made with 💜 by the MAPIE team · BSD-3-Clause License
</div>
