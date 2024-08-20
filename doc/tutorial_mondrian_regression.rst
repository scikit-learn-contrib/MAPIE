.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    from mapie.metrics import regression_coverage_score_v2
    from mapie.mondrian import MondrianCP
    from mapie.regression import MapieRegressor
    
    %load_ext autoreload
    %autoreload 2

.. code:: ipython3

    # Create 1D regression dataset with sinusoidual function between 0 and 10 
    n_points = 100000
    np.random.seed(0)
    X = np.linspace(0, 10, n_points).reshape(-1, 1)
    group_size = n_points // 10
    groups_list = []
    for i in range(10):
        groups_list.append(np.array([i] * group_size))
    groups = np.concatenate(groups_list)
    
    noise_0_1 = np.random.normal(0, 0.1, group_size)
    noise_1_2 = np.random.normal(0, 0.5, group_size)
    noise_2_3 = np.random.normal(0, 1, group_size)
    noise_3_4 = np.random.normal(0, .4, group_size)
    noise_4_5 = np.random.normal(0, .2, group_size)
    noise_5_6 = np.random.normal(0, .3, group_size)
    noise_6_7 = np.random.normal(0, .6, group_size)
    noise_7_8 = np.random.normal(0, .7, group_size)
    noise_8_9 = np.random.normal(0, .8, group_size)
    noise_9_10 = np.random.normal(0, .9, group_size)
    
    y = np.concatenate(
        [
            np.sin(X[groups == 0, 0] * 2) + noise_0_1,
            np.sin(X[groups == 1, 0] * 2) + noise_1_2,
            np.sin(X[groups == 2, 0] * 2) + noise_2_3,
            np.sin(X[groups == 3, 0] * 2) + noise_3_4,
            np.sin(X[groups == 4, 0] * 2) + noise_4_5,
            np.sin(X[groups == 5, 0] * 2) + noise_5_6,
            np.sin(X[groups == 6, 0] * 2) + noise_6_7,
            np.sin(X[groups == 7, 0] * 2) + noise_7_8,
            np.sin(X[groups == 8, 0] * 2) + noise_8_9,
            np.sin(X[groups == 9, 0] * 2) + noise_9_10,
        ], axis=0
    )
    


.. code:: ipython3

    plt.scatter(X, y, c=groups)
    plt.show()



.. image:: tutorial_mondrian_regression_files/tutorial_mondrian_regression_2_0.png


.. code:: ipython3

    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    groups_train_temp, groups_test, _, _ = train_test_split(groups, y, test_size=0.2, random_state=0)
    X_cal, X_train, y_cal, y_train = train_test_split(X_train_temp, y_train_temp, test_size=0.5, random_state=0)
    groups_cal, groups_train, _, _ = train_test_split(groups_train_temp, y_train_temp, test_size=0.5, random_state=0)

.. code:: ipython3

    X_train.shape, y_train.shape, groups_train.shape




.. parsed-literal::

    ((40000, 1), (40000,), (40000,))



.. code:: ipython3

    f, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(X_train, y_train, c=groups_train)
    ax[0].set_title("Train set")
    ax[1].scatter(X_cal, y_cal, c=groups_cal)
    ax[1].set_title("Calibration set")
    ax[2].scatter(X_test, y_test, c=groups_test)
    ax[2].set_title("Test set")
    plt.show()



.. image:: tutorial_mondrian_regression_files/tutorial_mondrian_regression_5_0.png


.. code:: ipython3

    print("Training set size: ", X_train.shape[0])
    print("Calibration set size: ", X_cal.shape[0])
    print("Test set size: ", X_test.shape[0])


.. parsed-literal::

    Training set size:  40000
    Calibration set size:  40000
    Test set size:  20000


.. code:: ipython3

    # Fit a random forest regressor
    
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)





.. raw:: html

    <style>#sk-container-id-1 {
      /* Definition of color scheme common for light and dark mode */
      --sklearn-color-text: black;
      --sklearn-color-line: gray;
      /* Definition of color scheme for unfitted estimators */
      --sklearn-color-unfitted-level-0: #fff5e6;
      --sklearn-color-unfitted-level-1: #f6e4d2;
      --sklearn-color-unfitted-level-2: #ffe0b3;
      --sklearn-color-unfitted-level-3: chocolate;
      /* Definition of color scheme for fitted estimators */
      --sklearn-color-fitted-level-0: #f0f8ff;
      --sklearn-color-fitted-level-1: #d4ebff;
      --sklearn-color-fitted-level-2: #b3dbfd;
      --sklearn-color-fitted-level-3: cornflowerblue;
    
      /* Specific color for light theme */
      --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
      --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-icon: #696969;
    
      @media (prefers-color-scheme: dark) {
        /* Redefinition of color scheme for dark theme */
        --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
        --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-icon: #878787;
      }
    }
    
    #sk-container-id-1 {
      color: var(--sklearn-color-text);
    }
    
    #sk-container-id-1 pre {
      padding: 0;
    }
    
    #sk-container-id-1 input.sk-hidden--visually {
      border: 0;
      clip: rect(1px 1px 1px 1px);
      clip: rect(1px, 1px, 1px, 1px);
      height: 1px;
      margin: -1px;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }
    
    #sk-container-id-1 div.sk-dashed-wrapped {
      border: 1px dashed var(--sklearn-color-line);
      margin: 0 0.4em 0.5em 0.4em;
      box-sizing: border-box;
      padding-bottom: 0.4em;
      background-color: var(--sklearn-color-background);
    }
    
    #sk-container-id-1 div.sk-container {
      /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
         but bootstrap.min.css set `[hidden] { display: none !important; }`
         so we also need the `!important` here to be able to override the
         default hidden behavior on the sphinx rendered scikit-learn.org.
         See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
      display: inline-block !important;
      position: relative;
    }
    
    #sk-container-id-1 div.sk-text-repr-fallback {
      display: none;
    }
    
    div.sk-parallel-item,
    div.sk-serial,
    div.sk-item {
      /* draw centered vertical line to link estimators */
      background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
      background-size: 2px 100%;
      background-repeat: no-repeat;
      background-position: center center;
    }
    
    /* Parallel-specific style estimator block */
    
    #sk-container-id-1 div.sk-parallel-item::after {
      content: "";
      width: 100%;
      border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
      flex-grow: 1;
    }
    
    #sk-container-id-1 div.sk-parallel {
      display: flex;
      align-items: stretch;
      justify-content: center;
      background-color: var(--sklearn-color-background);
      position: relative;
    }
    
    #sk-container-id-1 div.sk-parallel-item {
      display: flex;
      flex-direction: column;
    }
    
    #sk-container-id-1 div.sk-parallel-item:first-child::after {
      align-self: flex-end;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:last-child::after {
      align-self: flex-start;
      width: 50%;
    }
    
    #sk-container-id-1 div.sk-parallel-item:only-child::after {
      width: 0;
    }
    
    /* Serial-specific style estimator block */
    
    #sk-container-id-1 div.sk-serial {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--sklearn-color-background);
      padding-right: 1em;
      padding-left: 1em;
    }
    
    
    /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
    clickable and can be expanded/collapsed.
    - Pipeline and ColumnTransformer use this feature and define the default style
    - Estimators will overwrite some part of the style using the `sk-estimator` class
    */
    
    /* Pipeline and ColumnTransformer style (default) */
    
    #sk-container-id-1 div.sk-toggleable {
      /* Default theme specific background. It is overwritten whether we have a
      specific estimator or a Pipeline/ColumnTransformer */
      background-color: var(--sklearn-color-background);
    }
    
    /* Toggleable label */
    #sk-container-id-1 label.sk-toggleable__label {
      cursor: pointer;
      display: block;
      width: 100%;
      margin-bottom: 0;
      padding: 0.5em;
      box-sizing: border-box;
      text-align: center;
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:before {
      /* Arrow on the left of the label */
      content: "▸";
      float: left;
      margin-right: 0.25em;
      color: var(--sklearn-color-icon);
    }
    
    #sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
      color: var(--sklearn-color-text);
    }
    
    /* Toggleable content - dropdown */
    
    #sk-container-id-1 div.sk-toggleable__content {
      max-height: 0;
      max-width: 0;
      overflow: hidden;
      text-align: left;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content pre {
      margin: 0.2em;
      border-radius: 0.25em;
      color: var(--sklearn-color-text);
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-toggleable__content.fitted pre {
      /* unfitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
      /* Expand drop-down */
      max-height: 200px;
      max-width: 100%;
      overflow: auto;
    }
    
    #sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
      content: "▾";
    }
    
    /* Pipeline/ColumnTransformer-specific style */
    
    #sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator-specific style */
    
    /* Colorize estimator box */
    #sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    #sk-container-id-1 div.sk-label label.sk-toggleable__label,
    #sk-container-id-1 div.sk-label label {
      /* The background is the default theme color */
      color: var(--sklearn-color-text-on-default-background);
    }
    
    /* On hover, darken the color of the background */
    #sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    /* Label box, darken color on hover, fitted */
    #sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator label */
    
    #sk-container-id-1 div.sk-label label {
      font-family: monospace;
      font-weight: bold;
      display: inline-block;
      line-height: 1.2em;
    }
    
    #sk-container-id-1 div.sk-label-container {
      text-align: center;
    }
    
    /* Estimator-specific */
    #sk-container-id-1 div.sk-estimator {
      font-family: monospace;
      border: 1px dotted var(--sklearn-color-border-box);
      border-radius: 0.25em;
      box-sizing: border-box;
      margin-bottom: 0.5em;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    /* on hover */
    #sk-container-id-1 div.sk-estimator:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-1 div.sk-estimator.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Specification for estimator info (e.g. "i" and "?") */
    
    /* Common style for "i" and "?" */
    
    .sk-estimator-doc-link,
    a:link.sk-estimator-doc-link,
    a:visited.sk-estimator-doc-link {
      float: right;
      font-size: smaller;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1em;
      height: 1em;
      width: 1em;
      text-decoration: none !important;
      margin-left: 1ex;
      /* unfitted */
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
      color: var(--sklearn-color-unfitted-level-1);
    }
    
    .sk-estimator-doc-link.fitted,
    a:link.sk-estimator-doc-link.fitted,
    a:visited.sk-estimator-doc-link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    div.sk-estimator:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover,
    div.sk-label-container:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover,
    div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    /* Span, style for the box shown on hovering the info icon */
    .sk-estimator-doc-link span {
      display: none;
      z-index: 9999;
      position: relative;
      font-weight: normal;
      right: .2ex;
      padding: .5ex;
      margin: .5ex;
      width: min-content;
      min-width: 20ex;
      max-width: 50ex;
      color: var(--sklearn-color-text);
      box-shadow: 2pt 2pt 4pt #999;
      /* unfitted */
      background: var(--sklearn-color-unfitted-level-0);
      border: .5pt solid var(--sklearn-color-unfitted-level-3);
    }
    
    .sk-estimator-doc-link.fitted span {
      /* fitted */
      background: var(--sklearn-color-fitted-level-0);
      border: var(--sklearn-color-fitted-level-3);
    }
    
    .sk-estimator-doc-link:hover span {
      display: block;
    }
    
    /* "?"-specific style due to the `<a>` HTML tag */
    
    #sk-container-id-1 a.estimator_doc_link {
      float: right;
      font-size: 1rem;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1rem;
      height: 1rem;
      width: 1rem;
      text-decoration: none;
      /* unfitted */
      color: var(--sklearn-color-unfitted-level-1);
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    #sk-container-id-1 a.estimator_doc_link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    #sk-container-id-1 a.estimator_doc_link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
    }
    </style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestRegressor()</pre></div> </div></div></div></div>



.. code:: ipython3

    # Plot the prediction of the random forest regressor as a line
    
    y_pred = rf.predict(X_test)
    # plt.scatter(X_test, y_test, label="True")
    
    #Sort the test set and the prediction to plot them as a line
    sort_idx = np.argsort(X_test[:, 0])
    plt.plot(X_test[sort_idx], y_pred[sort_idx], label="Prediction")
    
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x1380f0760>




.. image:: tutorial_mondrian_regression_files/tutorial_mondrian_regression_8_1.png


.. code:: ipython3

    mapie_regressor = MapieRegressor(rf, cv="prefit")
    mondrian_regressor = MondrianCP(MapieRegressor(rf, cv="prefit"))

.. code:: ipython3

    mapie_regressor.fit(X_cal, y_cal)
    mondrian_regressor.fit(X_cal, y_cal, groups=groups_cal)




.. raw:: html

    <style>#sk-container-id-2 {
      /* Definition of color scheme common for light and dark mode */
      --sklearn-color-text: black;
      --sklearn-color-line: gray;
      /* Definition of color scheme for unfitted estimators */
      --sklearn-color-unfitted-level-0: #fff5e6;
      --sklearn-color-unfitted-level-1: #f6e4d2;
      --sklearn-color-unfitted-level-2: #ffe0b3;
      --sklearn-color-unfitted-level-3: chocolate;
      /* Definition of color scheme for fitted estimators */
      --sklearn-color-fitted-level-0: #f0f8ff;
      --sklearn-color-fitted-level-1: #d4ebff;
      --sklearn-color-fitted-level-2: #b3dbfd;
      --sklearn-color-fitted-level-3: cornflowerblue;
    
      /* Specific color for light theme */
      --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
      --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-icon: #696969;
    
      @media (prefers-color-scheme: dark) {
        /* Redefinition of color scheme for dark theme */
        --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
        --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-icon: #878787;
      }
    }
    
    #sk-container-id-2 {
      color: var(--sklearn-color-text);
    }
    
    #sk-container-id-2 pre {
      padding: 0;
    }
    
    #sk-container-id-2 input.sk-hidden--visually {
      border: 0;
      clip: rect(1px 1px 1px 1px);
      clip: rect(1px, 1px, 1px, 1px);
      height: 1px;
      margin: -1px;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }
    
    #sk-container-id-2 div.sk-dashed-wrapped {
      border: 1px dashed var(--sklearn-color-line);
      margin: 0 0.4em 0.5em 0.4em;
      box-sizing: border-box;
      padding-bottom: 0.4em;
      background-color: var(--sklearn-color-background);
    }
    
    #sk-container-id-2 div.sk-container {
      /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
         but bootstrap.min.css set `[hidden] { display: none !important; }`
         so we also need the `!important` here to be able to override the
         default hidden behavior on the sphinx rendered scikit-learn.org.
         See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
      display: inline-block !important;
      position: relative;
    }
    
    #sk-container-id-2 div.sk-text-repr-fallback {
      display: none;
    }
    
    div.sk-parallel-item,
    div.sk-serial,
    div.sk-item {
      /* draw centered vertical line to link estimators */
      background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
      background-size: 2px 100%;
      background-repeat: no-repeat;
      background-position: center center;
    }
    
    /* Parallel-specific style estimator block */
    
    #sk-container-id-2 div.sk-parallel-item::after {
      content: "";
      width: 100%;
      border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
      flex-grow: 1;
    }
    
    #sk-container-id-2 div.sk-parallel {
      display: flex;
      align-items: stretch;
      justify-content: center;
      background-color: var(--sklearn-color-background);
      position: relative;
    }
    
    #sk-container-id-2 div.sk-parallel-item {
      display: flex;
      flex-direction: column;
    }
    
    #sk-container-id-2 div.sk-parallel-item:first-child::after {
      align-self: flex-end;
      width: 50%;
    }
    
    #sk-container-id-2 div.sk-parallel-item:last-child::after {
      align-self: flex-start;
      width: 50%;
    }
    
    #sk-container-id-2 div.sk-parallel-item:only-child::after {
      width: 0;
    }
    
    /* Serial-specific style estimator block */
    
    #sk-container-id-2 div.sk-serial {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--sklearn-color-background);
      padding-right: 1em;
      padding-left: 1em;
    }
    
    
    /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
    clickable and can be expanded/collapsed.
    - Pipeline and ColumnTransformer use this feature and define the default style
    - Estimators will overwrite some part of the style using the `sk-estimator` class
    */
    
    /* Pipeline and ColumnTransformer style (default) */
    
    #sk-container-id-2 div.sk-toggleable {
      /* Default theme specific background. It is overwritten whether we have a
      specific estimator or a Pipeline/ColumnTransformer */
      background-color: var(--sklearn-color-background);
    }
    
    /* Toggleable label */
    #sk-container-id-2 label.sk-toggleable__label {
      cursor: pointer;
      display: block;
      width: 100%;
      margin-bottom: 0;
      padding: 0.5em;
      box-sizing: border-box;
      text-align: center;
    }
    
    #sk-container-id-2 label.sk-toggleable__label-arrow:before {
      /* Arrow on the left of the label */
      content: "▸";
      float: left;
      margin-right: 0.25em;
      color: var(--sklearn-color-icon);
    }
    
    #sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
      color: var(--sklearn-color-text);
    }
    
    /* Toggleable content - dropdown */
    
    #sk-container-id-2 div.sk-toggleable__content {
      max-height: 0;
      max-width: 0;
      overflow: hidden;
      text-align: left;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-2 div.sk-toggleable__content.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-2 div.sk-toggleable__content pre {
      margin: 0.2em;
      border-radius: 0.25em;
      color: var(--sklearn-color-text);
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-2 div.sk-toggleable__content.fitted pre {
      /* unfitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    #sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
      /* Expand drop-down */
      max-height: 200px;
      max-width: 100%;
      overflow: auto;
    }
    
    #sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
      content: "▾";
    }
    
    /* Pipeline/ColumnTransformer-specific style */
    
    #sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator-specific style */
    
    /* Colorize estimator box */
    #sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    #sk-container-id-2 div.sk-label label.sk-toggleable__label,
    #sk-container-id-2 div.sk-label label {
      /* The background is the default theme color */
      color: var(--sklearn-color-text-on-default-background);
    }
    
    /* On hover, darken the color of the background */
    #sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    /* Label box, darken color on hover, fitted */
    #sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Estimator label */
    
    #sk-container-id-2 div.sk-label label {
      font-family: monospace;
      font-weight: bold;
      display: inline-block;
      line-height: 1.2em;
    }
    
    #sk-container-id-2 div.sk-label-container {
      text-align: center;
    }
    
    /* Estimator-specific */
    #sk-container-id-2 div.sk-estimator {
      font-family: monospace;
      border: 1px dotted var(--sklearn-color-border-box);
      border-radius: 0.25em;
      box-sizing: border-box;
      margin-bottom: 0.5em;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }
    
    #sk-container-id-2 div.sk-estimator.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }
    
    /* on hover */
    #sk-container-id-2 div.sk-estimator:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }
    
    #sk-container-id-2 div.sk-estimator.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }
    
    /* Specification for estimator info (e.g. "i" and "?") */
    
    /* Common style for "i" and "?" */
    
    .sk-estimator-doc-link,
    a:link.sk-estimator-doc-link,
    a:visited.sk-estimator-doc-link {
      float: right;
      font-size: smaller;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1em;
      height: 1em;
      width: 1em;
      text-decoration: none !important;
      margin-left: 1ex;
      /* unfitted */
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
      color: var(--sklearn-color-unfitted-level-1);
    }
    
    .sk-estimator-doc-link.fitted,
    a:link.sk-estimator-doc-link.fitted,
    a:visited.sk-estimator-doc-link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    div.sk-estimator:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover,
    div.sk-label-container:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover,
    div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    /* Span, style for the box shown on hovering the info icon */
    .sk-estimator-doc-link span {
      display: none;
      z-index: 9999;
      position: relative;
      font-weight: normal;
      right: .2ex;
      padding: .5ex;
      margin: .5ex;
      width: min-content;
      min-width: 20ex;
      max-width: 50ex;
      color: var(--sklearn-color-text);
      box-shadow: 2pt 2pt 4pt #999;
      /* unfitted */
      background: var(--sklearn-color-unfitted-level-0);
      border: .5pt solid var(--sklearn-color-unfitted-level-3);
    }
    
    .sk-estimator-doc-link.fitted span {
      /* fitted */
      background: var(--sklearn-color-fitted-level-0);
      border: var(--sklearn-color-fitted-level-3);
    }
    
    .sk-estimator-doc-link:hover span {
      display: block;
    }
    
    /* "?"-specific style due to the `<a>` HTML tag */
    
    #sk-container-id-2 a.estimator_doc_link {
      float: right;
      font-size: 1rem;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1rem;
      height: 1rem;
      width: 1rem;
      text-decoration: none;
      /* unfitted */
      color: var(--sklearn-color-unfitted-level-1);
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
    }
    
    #sk-container-id-2 a.estimator_doc_link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }
    
    /* On hover */
    #sk-container-id-2 a.estimator_doc_link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }
    
    #sk-container-id-2 a.estimator_doc_link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
    }
    </style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MondrianCP(mapie_estimator=MapieRegressor(cv=&#x27;prefit&#x27;,
                                              estimator=RandomForestRegressor()))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;MondrianCP<span class="sk-estimator-doc-link ">i<span>Not fitted</span></span></label><div class="sk-toggleable__content "><pre>MondrianCP(mapie_estimator=MapieRegressor(cv=&#x27;prefit&#x27;,
                                              estimator=RandomForestRegressor()))</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label  sk-toggleable__label-arrow ">mapie_estimator: MapieRegressor</label><div class="sk-toggleable__content "><pre>MapieRegressor(cv=&#x27;prefit&#x27;, estimator=RandomForestRegressor())</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label  sk-toggleable__label-arrow ">estimator: RandomForestRegressor</label><div class="sk-toggleable__content "><pre>RandomForestRegressor()</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator  sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label  sk-toggleable__label-arrow ">&nbsp;RandomForestRegressor<a class="sk-estimator-doc-link " rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html">?<span>Documentation for RandomForestRegressor</span></a></label><div class="sk-toggleable__content "><pre>RandomForestRegressor()</pre></div> </div></div></div></div></div></div></div></div></div></div></div></div></div></div>



.. code:: ipython3

    _, y_pss_split = mapie_regressor.predict(X_test, alpha=.1)
    _, y_pss_mondrian = mondrian_regressor.predict(X_test, groups=groups_test, alpha=.1)

.. code:: ipython3

    rf = RandomForestRegressor(
        n_estimators=100
    )
    rf.fit(X_train, y_train)
    mondrian_regressor = MondrianCP(
        MapieRegressor(rf, cv="prefit")
    )
    mondrian_regressor.fit(
        X_cal, y_cal,
        groups=groups_cal
    )
    _, y_pss_mondrian = mondrian_regressor.predict(
        X_test, groups=groups_test, alpha=.1
    )

.. code:: ipython3

    # Plot the prediction of the random forest regressor as a line with the prediction intervals
    
    # plt.scatter(X_test, y_test, label="True")
    sort_idx = np.argsort(X_test[:, 0])
    # plt.plot(X_test[sort_idx], y_pred[sort_idx], label="Prediction")
    plt.fill_between(X_test[sort_idx].flatten(), y_pss_split[sort_idx, 0].flatten(), y_pss_split[sort_idx, 1].flatten(), alpha=0.3, label="Split")
    plt.fill_between(X_test[sort_idx].flatten(), y_pss_mondrian[sort_idx, 0].flatten(), y_pss_mondrian[sort_idx, 1].flatten(), alpha=0.3, label="Mondrian")
    plt.legend()
    plt.show()



.. image:: tutorial_mondrian_regression_files/tutorial_mondrian_regression_13_0.png


.. code:: ipython3

    # plot coverage by groups with both methods
    coverages = {}
    for group in np.unique(groups_test):
        coverages[group] = {}
        coverages[group]["split"] = regression_coverage_score_v2(y_test[groups_test == group], y_pss_split[groups_test == group])
        coverages[group]["mondrian"] = regression_coverage_score_v2(y_test[groups_test == group], y_pss_mondrian[groups_test == group])

.. code:: ipython3

    # Plot the coverage by groups, plot both methods side by side
    plt.bar(np.arange(len(coverages)) * 2, [float(coverages[group]["split"]) for group in coverages], label="Split")
    plt.bar(np.arange(len(coverages)) * 2 + 1, [float(coverages[group]["mondrian"]) for group in coverages], label="Mondrian")
    plt.xticks(np.arange(len(coverages)) * 2 + .5, [f"Group {group}" for group in coverages], rotation=45)
    plt.hlines(0.9, -1, 21, label="90% coverage", color="black", linestyle="--")
    plt.ylabel("Coverage")
    
    #put legend outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


.. parsed-literal::

    /var/folders/7d/cdjx7c6d3xx42wdw5bnrmmb80000gn/T/ipykernel_90633/2054907134.py:2: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      plt.bar(np.arange(len(coverages)) * 2, [float(coverages[group]["split"]) for group in coverages], label="Split")
    /var/folders/7d/cdjx7c6d3xx42wdw5bnrmmb80000gn/T/ipykernel_90633/2054907134.py:3: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      plt.bar(np.arange(len(coverages)) * 2 + 1, [float(coverages[group]["mondrian"]) for group in coverages], label="Mondrian")




.. parsed-literal::

    <matplotlib.legend.Legend at 0x13847b970>




.. image:: tutorial_mondrian_regression_files/tutorial_mondrian_regression_15_2.png


