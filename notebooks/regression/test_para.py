import numpy as np

def testing_para_CQR(iters, n_samples=1000, alphas=0.1, funct=x_sinx, estimator_name="GradientBoostingRegressor", symmetry=True):
    """
    Testing how changing the number of samples or the value of alpha affects both coverage and interval width.
    """
    random_state = 1
    if isinstance(n_samples, list) and isinstance(alphas, list):
        raise ValueError("Either n_samples or alphas needs to be a single value.")
    if isinstance(n_samples, int) and isinstance(alphas, float):
        raise ValueError("Either n_samples or alphas needs to be a list.")
    
    list_values = []
    list_initial_coverage = []
    list_mapie_coverage = []
    list_initial_width = []
    list_mapie_width = []
    list_scores = []

    if isinstance(alphas, float):
        name = "n_samples"
        values = n_samples
    else:
        name = "alphas"
        values = alphas

    for value in values:
        for i in range(iters):
            random_state = np.random.randint(0, 1000)
            list_values.append(value)
            if isinstance(alphas, float):
                    X_train_, X_calib_, y_train_, y_calib_, X_test, y_test, _ = get_data(funct, n_samples=value, random_state=random_state, data_name="paper_reproduction")
            else:
                    X_train_, X_calib_, y_train_, y_calib_, X_test, y_test, _ = get_data(funct, n_samples=n_samples, random_state=random_state, data_name="paper_reproduction")
            try:
                estimator=get_estimator(estimator_name)
                if isinstance(n_samples, int):
                    mapie_reg = MapieQuantileRegressor(
                            estimator=estimator,
                            alpha=value,
                    )
                    y_pred_qr = quantile_regression(estimator, X_train_, y_train_, X_test, y_test, alpha=value)
                else:
                    mapie_reg = MapieQuantileRegressor(
                            estimator=estimator,
                            alpha=alphas,
                    )
                    y_pred_qr = quantile_regression(estimator, X_train_, y_train_, X_test, y_test, alpha=alphas)

                mapie_reg.fit(X_train_, y_train_, X_calib=X_calib_, y_calib=y_calib_)
                y_pred, y_pis, = mapie_reg.predict(X_test, symmetry=symmetry)

                list_initial_coverage.append(regression_coverage_score(y_test, y_pred_qr[0], y_pred_qr[1]))
                list_mapie_coverage.append(regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]))
                list_initial_width.append(regression_mean_width_score(y_pred_qr[0], y_pred_qr[1]))
                list_mapie_width.append(regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0]))

                list_scores.append(mapie_reg.conformity_scores_)
            except:
                list_initial_coverage.append(np.nan)
                list_mapie_coverage.append(np.nan)
                list_initial_width.append(np.nan)
                list_mapie_width.append(np.nan)

    return pd.DataFrame(zip(list_values, list_initial_coverage, list_mapie_coverage, list_initial_width, list_mapie_width), columns=[name, "init_cov", "mapie_cov", "init_piw", "mapie_piw"])