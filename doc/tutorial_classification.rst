.. title:: Tutorial : contents

.. _tutorial_classification:

========
Tutorial
========

In this tutorial, we compare the prediction sets estimating by Mapieclassifier in a two-dimensional dataset with three labels. The distribution of the data is a bivariate normal with diagonal covariance matrices for each label. 

Throughout this tutorial, we will answer the following questions:

How do the number of class in the prediction sets vary according to the values ​​of alpha with the Conformal Prediction method ?

1. Conformal Prediction method using the softmax score of the true label
=====================================================================


* First we split the dataset in train/calib/test and the Model is fitted in the training set.
* We set the conformal score Si = f(Xi)Yi the softmax output of the true class for each sample in the calibration set.
* Then we define q as being the (n + 1) (α) ⌉ / n previous quantile of S1, ..., Sn (this is essentially the quantile α, but with a small correction). 
* Finally, for a new test data point (where Xn + 1 is known but Yn + 1 is not), create a prediction set T (Xn + 1) = {y: f (Xn + 1) y > q} which includes all the classes with a sufficiently high softmax output.

.. code-block:: python

   import numpy as np
   # Create training set from multivariate normal distribution
   centers = [(0, 3.5), (-2, 0), (2, 0)]
   covs = [[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[5, 0], [0, 1]]]
   x_min, x_max, y_min, y_max, step = -5, 7, -5, 7, 0.1
   n_samples = 500
   alphas = [0.2, 0.1, 0.05]
   X_train = np.zeros((3*n_samples, 2))
   i = 0
   for center, cov in zip(centers, covs):
       (
           X_train[i*n_samples:(i+1)*n_samples, 0],
           X_train[i*n_samples:(i+1)*n_samples, 1]
       ) = np.random.multivariate_normal(center, cov, n_samples).T
       i += 1
   y_train = np.stack([i for i in range(3) for _ in range(n_samples)]).ravel()

   # Create test from (x, y) coordinates
   X_test = np.stack([
       [x, y]
       for x in np.arange(x_min, x_max, step)
       for y in np.arange(x_min, x_max, step)
   ])

Let's see our training data

.. code-block:: python

   import matplotlib.pyplot as plt
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
   y_train_col = [colors[int(i)] for _, i in enumerate(y_train)]
   fig = plt.figure()
   plt.scatter(
       X_train[:, 0],
       X_train[:, 1],
       color=y_train_col,
       marker='o',
       s=10,
       edgecolor='k'
   )
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.show()

.. image:: images/tuto_classification_1.jpeg
    :align: center

We fit our training data with a Gaussian Naive Base estimator. And then we apply mapieclassifier to the estimator indicating that it has already been fitted.
We then estimate the prediction sets with differents alphas with a
``fit`` and ``predict`` process. 

.. code-block:: python

   from sklearn.naive_bayes import GaussianNB
   from mapie.classification import MapieClassifier
   from mapie.metrics import classification_coverage_score
   clf = GaussianNB().fit(X_train, y_train)
   nb_y_pred = clf.predict(X_test)
   nb_y_pred_proba = clf.predict_proba(X_test)
   nb_y_pred_proba_max = np.max(nb_y_pred_proba, axis=1)
   nb_mapie = MapieClassifier(estimator=clf, cv="prefit")
   nb_mapie.fit(X_train, y_train)
   nb_y_pred_mapie, nb_y_ps_mapie = nb_mapie.predict(X_test, alpha=alphas)


* y_pred_mapie: represents the prediction in the test set with the estimator.
* y_ps_mapie: the prediction sets with mapie.

.. code-block:: python

   def plot_scores(n, scores, quantiles):    
       print(quantiles)   
       colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
       fig = plt.figure()
       plt.hist(scores, bins='auto')
       i=0         
       for quantile in quantiles:
           plt.vlines(x = quantile, ymin=0, ymax=600, color = colors[i], linestyles = 'dashed',label='test') 
           i=i+1
       plt.title("Distribution of scores")
       plt.legend(alphas)
       plt.xlabel("scores")
       plt.ylabel("count")
       plt.show()

   def plot_result(alphas, y_pred_mapie, y_ps_mapie):
       tab10 = plt.cm.get_cmap('Purples', 4)
       y_pred_col = [colors[int(i)] for _, i in enumerate(y_pred_mapie)]
       fig, axs = plt.subplots(1, 4, figsize=(20, 4))
       axs[0].scatter(
           X_test[:, 0],
           X_test[:, 1],
           color=y_pred_col,
           marker='.',
           s=10,
           alpha=0.4
       )
       axs[0].set_title("Predicted labels")
       for i, alpha in enumerate(alphas):
           y_pi_sums = y_ps_mapie[:, :, i].sum(axis=1)
           num_labels = axs[i+1].scatter(
               X_test[:, 0],
               X_test[:, 1],
               c=y_pi_sums,
               marker='.',
               s=10,
               alpha=1,
               cmap=tab10,
               vmin=0,
               vmax=3
           )
           cbar = plt.colorbar(num_labels, ax=axs[i+1])
           coverage= classification_coverage_score(y_pred_mapie,y_ps_mapie[:,:,i])
           axs[i+1].set_title(
               f"Number of labels for alpha={alpha}: ({1 - alpha:.3f}, {coverage:.3f})",
               fontsize=8
           )
       plt.show()

Let's see the distribution of the scores with the calculated quantiles.

.. code-block:: python

   scores = nb_mapie.scores_
   n = nb_mapie.n_samples_val_
   quantiles = nb_mapie.quantiles_ 
   plot_scores(n, scores, quantiles)

.. image:: images/tuto_classification_2.jpeg
    :align: center

We will now compare the differences between the prediction sets of the different values ​​of alpha.

.. code-block:: python

   plot_result(alphas,nb_y_pred_mapie,  nb_y_ps_mapie)

.. image:: images/tuto_classification_3.jpeg
    :align: center

When the class coverage is not large enough, the prediction sets can be empty
when the model is uncertain at the border between two labels. The null region
disappears for larger class coverages but ambiguous classification regions
arise with several labels included in the prediction sets.