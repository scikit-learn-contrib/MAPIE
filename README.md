[![GitHub Actions](https://github.com/scikit-learn-contrib/MAPIE/actions/workflows/test.yml/badge.svg)](https://github.com/scikit-learn-contrib/MAPIE/actions)
[![Codecov](https://codecov.io/gh/scikit-learn-contrib/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1)](https://codecov.io/gh/scikit-learn-contrib/MAPIE)
[![Documentation Status](https://readthedocs.org/projects/mapie/badge/?version=stable)](https://mapie.readthedocs.io/en/stable/?badge=stable)
[![License](https://img.shields.io/github/license/scikit-learn-contrib/MAPIE)](https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/mapie)](https://pypi.org/project/mapie/)
[![PyPI](https://img.shields.io/pypi/v/mapie)](https://pypi.org/project/mapie/)
[![Downloads](https://img.shields.io/pypi/dm/mapie)](https://pypistats.org/packages/mapie)
[![Conda](https://img.shields.io/conda/vn/conda-forge/mapie)](https://anaconda.org/conda-forge/mapie)
[![Release](https://img.shields.io/github/v/release/scikit-learn-contrib/mapie)](https://github.com/scikit-learn-contrib/MAPIE/releases)
[![Commits](https://img.shields.io/github/commits-since/scikit-learn-contrib/mapie/latest)](https://github.com/scikit-learn-contrib/MAPIE/commits/master)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/scikit-learn-contrib/MAPIE)

<p align="center">
  <img src="https://github.com/scikit-learn-contrib/MAPIE/raw/master/doc/images/mapie_logo_nobg_cut.png" width="400" alt="MAPIE logo">
</p>

# MAPIE - Model Agnostic Prediction Interval Estimator

🚀 MAPIE in 2026 🚀 New features have been implemented, starting with the application of **risk control** to emerging use cases such as **LLM-as-Judge** and **image segmentation**. In addition, **exchangeability tests** have been introduced to help users verify when MAPIE can be legitimately applied. Also, new **adaptive** conformal prediction methods have been added. Finally, the documentation has been updated with a new design!

🎉 MAPIE in 2025 🎉 MAPIE v1 is live! You're seeing the documentation of this new version, which introduces major changes to the API. Extensive release notes are available in the [documentation](https://mapie.readthedocs.io/en/stable/getting-started/v1-release-notes/). You can switch to the documentation of previous versions using the Read the Docs version menu.

See [GitHub Releases](https://github.com/scikit-learn-contrib/MAPIE/releases) and [HISTORY.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.md) for up-to-date details on the new features.

**MAPIE** is an open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.

<p align="center">
  <img src="https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/refs/heads/master/doc/images/educational_visual.png" width="500" alt="Educational visual for MAPIE">
  <br>
  <sub>Image credits: Cemrecan Yurtman (portrait) and hogrmahmood (zebra-horse hybrid).</sub>
</p>

MAPIE allows you to:

- **Compute prediction intervals or prediction sets** for regression,
  classification, and time series by estimating your model uncertainty on a
  conformalization dataset.
- **Control risks** of more complex tasks such as multi-label classification
  and semantic segmentation in computer vision, providing probabilistic
  guarantees on metrics like recall and precision.
- Easily use **any model (scikit-learn, TensorFlow, PyTorch)** thanks to scikit-learn-compatible wrapper if needed. MAPIE is part of the scikit-learn-contrib ecosystem.

MAPIE relies notably on the fields of Conformal Prediction and Distribution-Free Inference. It implements **peer-reviewed** algorithms that are  **model and use case agnostic** and possesses **theoretical guarantees** under minimal assumptions on the data and the model.

## 🛠 Requirements & installation

MAPIE runs on:

- Python >=3.9
- NumPy >=1.23
- scikit-learn >=1.4

MAPIE can be installed in different ways:

```sh
$ pip install mapie  # installation via `pip`
$ conda install -c conda-forge mapie  # or via `conda`
$ pip install git+https://github.com/scikit-learn-contrib/MAPIE  # or directly from the github repository
```

## ⚡ Quickstart and documentation

Below are two simple examples from [our documentation](https://mapie.readthedocs.io/en/latest/) that show how MAPIE is used in a regression setting and a classification setting:

- [Uncertainty quantification for a regression task](https://mapie.readthedocs.io/en/latest/generated/regression/1-quickstart/plot_toy_model/)

- [Uncertainty quantification for a classification task](https://mapie.readthedocs.io/en/latest/generated/classification/1-quickstart/plot_quickstart_classification/)

## 📝 Contributing

You are welcome to propose and contribute new ideas.
We encourage you to [open an issue](https://github.com/scikit-learn-contrib/MAPIE/issues) so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.
For more information on the contribution process, read our [contribution guidelines](https://github.com/scikit-learn-contrib/MAPIE/blob/master/CONTRIBUTING.md).

## 📚 License & citation

MAPIE is free and open-source software licensed under the [BSD-3-Clause license](https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE).

If you use MAPIE in your research, see the
[preferred citation, BibTeX, foundational references, and project affiliations](https://mapie.readthedocs.io/en/latest/text/about/citation/).
