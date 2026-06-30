# Metrics for Conditional Conformal Prediction - Theoretical Description

Conditional coverage diagnostics evaluate how prediction sets or intervals behave
across subpopulations, beyond their marginal coverage guarantee [^1].

### Coverage Gap (CovGap and WCovGap)

Measures how far empirical coverage is from the target coverage **inside
predefined groups** [^2]. For an evaluation set \(\{(x_i, y_i, g_i)\}_{i=1}^n\)
of size \(n\), let \(\alpha\) be the target miscoverage level,
\(\hat{C}(x_i)\) be the prediction set or interval for sample \(i\), \(G\) be
the set of observed groups, \(I_g = \{i: g_i = g\}\), and \(c_i\) be the
binary coverage indicator:

\[
c_i =
\begin{cases}
1, & y_i \in \hat{C}(x_i) \\
0, & \text{otherwise.}
\end{cases}
\]

For regression intervals, \(c_i = 1\) means
\(\hat{y}^{\text{low}}_i \leq y_i \leq \hat{y}^{\text{up}}_i\), where
\(\hat{y}^{\text{low}}_i\) and \(\hat{y}^{\text{up}}_i\) are the interval
endpoints. For classification sets, \(c_i = 1\) means
\(y_i \in \hat{C}(x_i)\).

The empirical coverage of group \(g\) is:

\[
\hat{Cov}_g = \frac{1}{|I_g|} \sum_{i \in I_g} c_i
\]

The unweighted coverage gap averages the absolute group-level deviations from
the target coverage \(1-\alpha\):

\[
\text{CovGap} = \frac{1}{|G|} \sum_{g \in G}
\left| \hat{Cov}_g - (1-\alpha) \right|
\]

The weighted coverage gap weights each group by its empirical sample
proportion:

\[
\text{WCovGap} = \sum_{g \in G} \frac{|I_g|}{n}
\left| \hat{Cov}_g - (1-\alpha) \right|
\]

CovGap gives small and large groups the same influence, which is useful when
each group is equally important. WCovGap gives more influence to larger groups,
which summarizes the average conditional coverage deviation over samples.

### Worst Slab Coverage (WSC)

Worst slab coverage evaluates conditional coverage over automatically selected
geometric subpopulations rather than predefined groups [^1][^3]. It searches
for regions of the feature space where the prediction sets or intervals cover
the smallest fraction of true labels.

For a feature vector \(x \in \mathbb{R}^p\), a unit direction
\(v \in \mathbb{S}^{p-1}\), and two thresholds \(a \leq b\), a slab is the
set of points whose projection onto \(v\) lies between \(a\) and \(b\):

\[
S(v, a, b) =
\left\{ x : a \leq v^\top x \leq b \right\}.
\]

At the population level, the worst slab coverage with minimum mass
\(\delta \in (0, 1)\) is:

\[
\text{WSC}_{\delta}
= \inf_{\substack{v \in \mathbb{S}^{p-1},\, a \leq b \\
\mathbb{P}(X \in S(v, a, b)) \geq \delta}}
\mathbb{P}\left(Y \in \hat{C}(X) \mid X \in S(v, a, b)\right).
\]

The parameter \(\delta\) prevents the metric from selecting slabs that are too
small to be meaningful. Larger values of \(\delta\) force the search to consider
larger subpopulations, while smaller values allow more localized diagnostics.

MAPIE estimates this quantity from an evaluation set. Let
\(\mathcal{V}_M = \{v_1, \ldots, v_M\}\) be \(M\) random directions sampled on
the unit sphere, where \(M\) corresponds to the `n_directions` parameter. For a
candidate slab, define:

\[
I(v, a, b) =
\left\{ i : a \leq v^\top x_i \leq b \right\}.
\]

The empirical estimator is:

\[
\widehat{\text{WSC}}_{\delta}
= \min_{\substack{v \in \mathcal{V}_M,\, a \leq b \\
|I(v, a, b)| \geq \lceil \delta n \rceil}}
\frac{1}{|I(v, a, b)|}
\sum_{i \in I(v, a, b)} c_i.
\]

In practice, for each sampled direction, MAPIE sorts the projected points and
searches contiguous intervals in that ordering containing at least
\(\lceil \delta n \rceil\) samples. The returned WSC value is the smallest
empirical coverage found among these slabs.

Unlike CovGap, WSC does not compare coverage to the target \(1-\alpha\) and does
not require user-defined groups. It returns a coverage value directly: values
well below the target coverage reveal a projected region of the feature space
where the prediction sets or intervals under-cover.

---

## References

[^1]: Braun, S., Holzmüller, D., Jordan, M. I., and Bach, F. "Conditional Coverage Diagnostics for Conformal Prediction." *arXiv:2512.11779*, 2025. [arXiv:2512.11779](https://arxiv.org/abs/2512.11779).
[^2]: Ding, T., Angelopoulos, A., Bates, S., Jordan, M., and Tibshirani, R. J. "Class-conditional conformal prediction with many classes." *NeurIPS*, 2023. [arXiv:2306.09335](https://arxiv.org/abs/2306.09335).
[^3]: Cauchois, M., Gupta, S., and Duchi, J. "Knowing What You Know: Valid and Validated Confidence Sets in Multiclass and Multilabel Prediction." *JMLR*, 2021. [JMLR 22(81)](https://www.jmlr.org/papers/v22/20-753.html).
