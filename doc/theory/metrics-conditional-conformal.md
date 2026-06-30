# Metrics for Conditional Conformal Prediction - Theoretical Description

Conditional coverage diagnostics evaluate how prediction sets or intervals behave
across subpopulations, beyond their marginal coverage guarantee [^1].

### Coverage Gap (CovGap and WCovGap)

Measures how far empirical coverage is from the target coverage **inside
predefined groups** [^2]. Let \(G\) be the set of observed groups,
\(I_g = \{i: g_i = g\}\), and \(c_i\) be the binary coverage indicator:

\[
c_i =
\begin{cases}
1, & y_i \in \hat{C}(x_i) \\
0, & \text{otherwise.}
\end{cases}
\]

For regression intervals, \(c_i = 1\) means
\(\hat{y}^{\text{low}}_i \leq y_i \leq \hat{y}^{\text{up}}_i\). For
classification sets, \(c_i = 1\) means \(y_i \in \hat{C}(x_i)\).

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

---

## References

[^1]: Braun, S., Holzmüller, D., Jordan, M. I., and Bach, F. "Conditional Coverage Diagnostics for Conformal Prediction." *arXiv:2512.11779*, 2025. [arXiv:2512.11779](https://arxiv.org/abs/2512.11779).
[^2]: Ding, T., Angelopoulos, A., Bates, S., Jordan, M., and Tibshirani, R. J. "Class-conditional conformal prediction with many classes." *NeurIPS*, 2023. [arXiv:2306.09335](https://arxiv.org/abs/2306.09335).
