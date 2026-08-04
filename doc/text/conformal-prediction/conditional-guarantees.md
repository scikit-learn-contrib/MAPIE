# Conditional Conformal Prediction — Theoretical Description

Standard Conformal Prediction provides marginal guarantees: the true label/value is in the prediction set/interval **marginally** on the test data. Many standard methods provide prediction intervals that do not depend on the specific data point and thus fail to capture heteroscedasticity (the variance of errors is not constant across a regression model's observations). Some algorithms in MAPIE provide intervals that adapt to specific data points, such as Conformalized Quantile Regression (CQR) or the Residual Normalized Score. However, their theoretical guarantees are still marginal.

Here we present a Conditional Conformal Prediction (CCP) method [^1]. This model-agnostic method can create adaptive prediction intervals and provide coverage guarantees for subgroups, rather than only marginal coverage.

In MAPIE, this method has several advantages:

- It is model-agnostic (it depends only on the predictions, unlike CQR).
- It can create highly adaptive intervals whose varying widths reflect model uncertainty.
- It provides coverage guarantees for all subgroups of interest, helping to avoid bias.
- It can incorporate prior knowledge about the data or the model.

However, we will also see its disadvantages:

- Its adaptivity depends on the feature map, which can be difficult to define.
- Inference is much slower than for other methods because an optimization problem is solved for each test point.

In summary, it can create more adaptive intervals than other methods, but finding the best settings can be difficult and computationally expensive.

---

## How does it work?

### Method's intuition

We recall that the standard split conformal prediction set is defined as

$$
\hat{C}_{\textrm{split}}(X_{n+1}) = \{y: S(X_{n+1}, y) \leq S^*\}
$$

with $S^*$ the quantile of the conformity scores evaluated on the calibration set, corresponding to the chosen confidence level.

One of the insights of the paper is that finding the quantile can be done as an intercept-only quantile regression using the pinball loss. Then, instead of using a fixed quantile, it becomes possible to use a function that estimates conditional quantiles of $Y | X$, i.e., replacing $S^*$ by a function $\hat{g}_{S(X_{n+1}, y)}(X_{n+1})$.

To be able to find the best function, while having some coverage guarantees, we should select this function inside some defined class of functions $\mathcal{F}$.

This method is motivated by the following equivalence:

$$
\begin{array}{c}
\mathbb{P}(Y_{n+1} \in \hat{C} \; | \; X_{n+1}=x) = 1 - \alpha, \quad \text{for all x} \\
\Longleftrightarrow \\
\mathbb{E} \left[ f(X_{n+1}) \mathbb{I} \left\{ Y_{n+1} \in \hat{C}(X_{n+1}) \right\} \right] = 0, \quad \text{for all measurable f} \\
\end{array}
$$

This is the equation corresponding to the perfect conditional coverage, which is theoretically impossible to obtain. Then, relaxing this objective by replacing "all measurable f" with "all f belonging to some class $\mathcal{F}$" seems a way to get close to the perfect conditional coverage.

### The method follows 3 steps (for the finite-dimensional setting)

1. Choose a class of functions. The simple approach is to choose a class of finite dimension $d \in \mathbb{N}$, here linear functions, using, for any $\Phi \; : \; \mathcal{X} \to \mathbb{R}^d$ (chosen by the user):

    $$
    \mathcal{F} = \left\{ \Phi (\cdot)^T \beta  :  \beta \in \mathbb{R}^d \right\}
    $$

2. Find the best function of this class by solving the following optimization problem:

    $$
    \hat{g}_S := \arg\min_{g \in \mathcal{F}} \; \frac{1}{n+1} \sum_{i=1}^n{l_{\alpha} (g(X_i), S_i)} \; + \frac{1}{n+1}l_{\alpha} (g(X_{n+1}), S)
    $$

    In practice, because computing the set defined below requires to fit $\hat{g}_S$ for all $S \in \mathbb{R}$, which appears to be intractable, a dual formulation of the optimization problem is solved instead.


3. We use this optimized function $\hat{g}_S$ to compute the prediction intervals:

    $$
    \hat{C}(X_{n+1}) = \{ y : S(X_{n+1}, \: y) \leq \hat{g}_{S(X_{n+1}, y)}(X_{n+1}) \}
    $$


### Coverage guarantees


Following these steps, we have the coverage guarantee, $\forall f \in \mathcal{F}$:

$$
\mathbb{P}_f(Y_{n+1} \in \hat{C}(X_{n+1})) \geq 1 - \alpha
$$

$$
\text{and} \quad \left | \mathbb{E} \left[ f(X_{n+1}) \left(\mathbb{I} \left\{ Y_{n+1} \in \hat{C}(X_{n+1}) \right\} - (1 - \alpha) \right) \right] \right |
\leq \frac{d}{n+1} \mathbb{E} \left[ \max_{1 \leq i \leq n+1} \left|f(X_i)\right| \right]
$$

Note: if we want to have a homogeneous coverage on some given groups in $\mathcal{G}$, we can use $\mathcal{F} = \{ x \mapsto \sum_{G \in \mathcal{G}} \; \beta_G \mathbb{I} \{ x \in G \} : \beta_G \in \mathbb{R} \}$, then we have $\forall G \in \mathcal{G}$:

$$
\begin{aligned}
1 - \alpha
&\leq \mathbb{P} \left( Y_{n+1} \in \hat{C}_M^{n+1}(X_{n+1}) \; | \; X_{n+1} \in G \right) \\
&\leq 1- \alpha + \frac{|\mathcal{G}|}{(n+1) \mathbb{P}(X_{n+1} \in G)} \\
&= 1- \alpha + \frac{\text{number of groups in } \mathcal{G}}{\text{number of samples of } \{X_i\} \text{ in } G}
\end{aligned}
$$

---

### Limitations of the current implementation

The original paper [^1] introduced two settings. For the first one, finite-dimensional shifts, coverage is guaranteed on the groups defined by the feature map. The second one, infinite-dimensional shifts, tackles any covariate shift and quantifies the coverage error, as an exact coverage guarantee is theoretically impossible in this case. Currently, MAPIE implements only the finite-dimensional setting. The infinite-dimensional case needs further optimization because it is very slow (e.g., reproducing Figure 5 of the paper takes dozens of compute hours).

## How to use it in practice?

### Creating a class of functions adapted to our needs

The following will provide some tips on how to use the method. For practical
examples, see the regression and classification examples using
`ConditionalSplitConformalRegressor` and `ConditionalSplitConformalClassifier`.

1. If you want to avoid bias across subgroups and ensure homogeneous coverage for
   those, you can add indicator functions corresponding to those groups in
   `feature_map`.

2. You can inject prior knowledge into the method through `feature_map` if you have
   information about the conformity score distribution (domains with different
   behavior, expected model uncertainty depending on a given feature, etc.).

3. Empirically test the obtained coverage on a test set to make sure that the
   expected coverage is achieved.

### Avoid miscoverage

- To guarantee marginal coverage, you need to have an intercept term in the
  $\Phi$ function (meaning, a feature equal to $1$ for all $X_i$).

- Keep the number of dimensions $d$ reasonable compared with the
  conformalization set size.

---

## References

[^1]: Isaac Gibbs, John J Cherian, Emmanuel J Candès, Conformal prediction with conditional guarantees, Journal of the Royal Statistical Society Series B: Statistical Methodology, Volume 87, Issue 4, September 2025, Pages 1100–1126, [https://doi.org/10.1093/jrsssb/qkaf008](https://doi.org/10.1093/jrsssb/qkaf008).
