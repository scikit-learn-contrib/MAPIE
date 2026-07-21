# Conditional Conformal Prediction — Theoretical Description

The Conditional Conformal Prediction (CCP) method [^1] is a model agnostic conformal prediction method which can create adaptative prediction intervals.

In MAPIE, this method has a lot of advantages:

- It is model agnostic (it doesn't depend on the model but only on the predictions, unlike CQR).
- It can create very adaptative intervals (with a varying width which truly reflects the model uncertainty).
- While providing coverage guarantee on all sub-groups of interest (avoiding biases).
- With the possibility to inject prior knowledge about the data or the model.

However, we will also see its disadvantages:

- The adaptativity depends on the feature map which can be difficult to define.
- The inference is much longer than for the other methods as an optimization process is solved for each test point.

To conclude, it can create more adaptative intervals than the other methods, but it can be difficult to find the best settings and can have a big computational time.

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

The original paper [^1] introduced two settings. For the first one, finite-dimensional shifts, coverage is guaranteed on the groups defined by the feature map. The second one, infinite-dimensional shifts, tackles any covariate shift and allows to quantify the coverage error, as an exact coverage guarantee is theoretically impossible in this case. In MAPIE, only the finite-dimensional setting is implemented currently. The infinite-dimensional case needs further optimization as it is very slow (e.g., dozens hours of compute to reproduce Figure 5 of the paper).

## How to use it in practice?

### Creating a class of functions adapted to our needs

The following will provide some tips on how to use the method. For practical
examples, see the regression and classification examples using
`ConditionalSplitConformalRegressor` and `ConditionalSplitConformalClassifier`.

1. If you want to avoid bias on sub-groups and ensure a homogeneous coverage on
   those, you can add indicator functions corresponding to those groups in
   `feature_map`.

2. You can inject prior knowledge in the method through `feature_map`, if you have
   information about the conformity scores distribution (domains with different
   behavior, expected model uncertainty depending on a given feature, etc.).

3. Empirically test the obtained coverage on a test set, to make sure that the
   expected coverage is achieved.

### Avoid miscoverage

- To guarantee marginal coverage, you need to have an intercept term in the
  $\Phi$ function (meaning, a feature equal to $1$ for all $X_i$).

- Keep the number of dimensions $d$ reasonable compared with the
  conformalization set size.

---

## References

[^1]: Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès (2023). *Conformal Prediction With Conditional Guarantees.* [arXiv:2305.12616](https://arxiv.org/abs/2305.12616).
