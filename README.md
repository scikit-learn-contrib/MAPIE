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

**An open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.**

🚀 MAPIE in 2026 🚀 New features have been implemented, starting with the application of **risk control** to emerging use cases such as **LLM-as-Judge** and **image segmentation**. In addition, **exchangeability tests** have been introduced to help users verify when MAPIE can be legitimately applied. Also, new **adaptive** conformal prediction methods have been added. Finally, the documentation has been updated with a new design!

🎉 MAPIE in 2025 🎉 MAPIE v1 is live! This new version introduces major changes to the API. Check out the [release notes](content/getting-started/v1-release-notes.md).

See [GitHub Releases](https://github.com/scikit-learn-contrib/MAPIE/releases) and [HISTORY.md](https://github.com/scikit-learn-contrib/MAPIE/blob/master/HISTORY.md) for up-to-date details on the new features.


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

## 🔍 References

1. Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World.* Springer Nature, 2022.
2. Angelopoulos, Anastasios N., and Stephen Bates. "Conformal prediction: A gentle introduction." *Foundations and Trends® in Machine Learning* 16.4 (2023): 494–591.
3. Barber, Rina Foygel, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." *Annals of Statistics* 49.1 (2021): 486–507.
4. Kim, Byol, Chen Xu, and Rina Barber. "Predictive inference is free with the jackknife+-after-bootstrap." *Advances in Neural Information Processing Systems* 33 (2020): 4138–4149.
5. Sadinle, Mauricio, Jing Lei, and Larry Wasserman. "Least ambiguous set-valued classifiers with bounded error levels." *Journal of the American Statistical Association* 114.525 (2019): 223–234.
6. Romano, Yaniv, Matteo Sesia, and Emmanuel Candès. "Classification with valid and adaptive coverage." *Advances in Neural Information Processing Systems* 33 (2020): 3581–3591.
7. Angelopoulos, Anastasios N., et al. "Uncertainty sets for image classifiers using conformal prediction." *International Conference on Learning Representations* (2021).
8. Romano, Yaniv, Evan Patterson, and Emmanuel Candès. "Conformalized quantile regression." *Advances in Neural Information Processing Systems* 32 (2019).
9. Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic time-series." *International Conference on Machine Learning.* PMLR, 2021.
10. Bates, Stephen, et al. "Distribution-free, risk-controlling prediction sets." *Journal of the ACM* 68.6 (2021): 1–34.
11. Angelopoulos, Anastasios N., Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. "Conformal Risk Control." (2022).
12. Angelopoulos, Anastasios N., Stephen Bates, Emmanuel J. Candès, et al. "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022).

## 📚 License & citation

MAPIE is free and open-source software licensed under the [BSD-3-Clause license](https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE).

If you use MAPIE in your research, please cite the main paper:

> Cordier, Thibault, et al. "Flexible and systematic uncertainty estimation with conformal prediction via the MAPIE library." *Conformal and Probabilistic Prediction with Applications.* PMLR, 2023.

```bibtex
@inproceedings{Cordier_Flexible_and_Systematic_2023,
    author = {Cordier, Thibault and Blot, Vincent and Lacombe, Louis and Morzadec, Thomas and Capitaine, Arnaud and Brunel, Nicolas},
    booktitle = {Conformal and Probabilistic Prediction with Applications},
    title = {{Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library}},
    year = {2023}
}
```

You can also cite the ICML workshop manuscript:

> Taquet, Vianney, et al. "MAPIE: an open-source library for distribution-free uncertainty quantification." *arXiv preprint arXiv:2207.12274* (2022).

```bibtex
@article{taquet2022mapie,
    title = {MAPIE: an open-source library for distribution-free uncertainty quantification},
    author = {Taquet, Vianney and Blot, Vincent and Morzadec, Thomas and Lacombe, Louis and Brunel, Nicolas},
    journal = {arXiv preprint arXiv:2207.12274},
    year = {2022}
}
```

## 🤝 Affiliations

MAPIE has been developed through a collaboration between Capgemini Invent,
Quantmetry, Michelin, ENS Paris-Saclay, and with the financial support from
Région Île-de-France and Confiance.ai.

<p>
  <a href="https://www.capgemini.com/about-us/who-we-are/our-brands/capgemini-invent/"><img src="https://www.capgemini.com/wp-content/themes/capgemini2020/assets/images/capgemini-invent.svg" height="35" width="140" alt="Capgemini Invent"></a>
  <a href="https://www.inria.fr/"><img src="https://www.inria.fr/themes/custom/inria/logo/logo.svg" height="35" width="140" alt="Inria"></a>
  <a href="https://p16.inria.fr/fr/"><img src="https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/doc/images/logo_P16.png" height="45" width="60" alt="Projet P16"></a>
  <a href="https://www.michelin.com/en/"><img src="https://agngnconpm.cloudimg.io/v7/https://dgaddcosprod.blob.core.windows.net/corporate-production/attachments/cls05tqdd9e0o0tkdghwi9m7n-clooe1x0c3k3x0tlu4cxi6dpn-bibendum-salut.full.png" height="50" width="45" alt="Michelin"></a>
  <a href="https://ens-paris-saclay.fr/en"><img src="https://ens-paris-saclay.fr/sites/default/files/ENSPS_UPSAY_logo_couleur_2.png" height="35" width="140" alt="ENS Paris-Saclay"></a>
  <a href="https://www.confiance.ai/"><img src="https://pbs.twimg.com/profile_images/1443838558549258264/EvWlv1Vq_400x400.jpg" height="45" width="45" alt="Confiance.ai"></a>
  <a href="https://www.iledefrance.fr/"><img src="https://www.iledefrance.fr/sites/default/files/logo/2024-02/logoGagnerok.svg" height="35" width="140" alt="Région Île-de-France"></a>
</p>
