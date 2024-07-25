.. title:: Calibrators : contents

.. _theoretical_description_calibrators:

###############
Calibrators
###############

In Mapie, the conformalisation step is done directly inside
:class:`mapie.regression.MapieRegressor` or :class:`mapie.classification.MapieClassifier`,
depending on the ``method`` argument.
However, when implementing the new CCP method, we decided to externalize the conformalisation
step into a new object named ``calibrator``, to have more freedom and possible customisation.

The new classes (for regression and classification), based on :class:`mapie.futur.split.SplitCP` have 3 steps:

1. ``fit_predictor``, which fit the sklearn estimator
2. ``fit_calibrator``, which do the conformalisation (calling ``calibrator.fit``)
3. ``predict``, which compute the predictions and call ``calibrator.predict`` to create the prediction intervals

Thus, the calibrators, based on :class:`mapie.calibrators.base.BaseCalibrator`,
must have the two methods: ``fit`` and ``predict``.

Mapie currently implements calibrators for the CCP method and the naive method,
but any method of comformal prediction can be implemented by the user as
a subclass of :class:`mapie.calibrators.base.BaseCalibrator`.

Example of naive Split CP:
----------------------------

For instance, the :class:`mapie.calibrators.StandardCalibrator` implements
the :ref:`naive split method<theoretical_description_regression_naive>`:

* ``.fit`` computes :math:`\hat{q}_{n, \alpha}^+`, the :math:`(1-\alpha)` quantile of the distribution
* ``.predict`` comptues the prediction intervals with: :math:`\hat{\mu}(X_{n+1}) \pm \hat{q}_{n, \alpha}^+`


The CCP calibrators:
---------------------
For the CCP method (see :ref:`theoretical description<theoretical_description_ccp>`),
:class:`mapie.calibrators.CCPCalibrator` implements:

* ``.fit`` solve the optimization problem (see :ref:`step 2<theoretical_description_ccp_control_steps>`) to find the optimal :math:`\hat{g}`
* ``.predict`` comptues the prediction intervals using :math:`\hat{g}` (see :ref:`step 3<theoretical_description_ccp_control_steps>`)

We just need a way to define our :math:`\Phi` function (see :ref:`step 1<theoretical_description_ccp_control_steps>`).

Multiple subclasses are implemented to facilitate the definition of the :math:`\Phi` function,
but other could be implemented by the user as a subclass of :class:`mapie.calibrators.ccp.CCPCalibrator`.

1. :class:`mapie.calibrators.CustomCCP`:

   This class allows to define by hand the :math:`\Phi` function, as a
   concatenation of other functions which create features of ``X`` (or potentially ``y_pred`` or any exogenous variable ``z``)
   
   It can also be used to concatenate other :class:`mapie.calibrators.CCPCalibrator` instances.

2. :class:`mapie.calibrators.PolynomialCCP`:

   It create some polynomial features of ``X`` (or potentially ``y_pred`` or any exogenous variable ``z``).
   It could be created by hand using `CustomCCP`, it is just a way simplify the creation of :math:`\Phi`.

2. :class:`mapie.calibrators.GaussianCCP`:

   It create gaussian kernels, as done in the method's paper :ref:`[1]<theoretical_description_calibrators_references>`.
   It samples random points from the :math:`\{ X_i \}_i`, then compute gaussian distances
   between each point and :math:`X_{n+1}` with a given standard deviation :math:`\sigma`
   (which can be optimized using cross-validation), following the formula:

   .. math::
     e^{-\frac{|X_{n+1} - point|^2}{2\sigma ^2}}


.. _theoretical_description_calibrators_references:

References
==========

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees", `arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.