def CQR_alphas(alphas, random_data, random_split, iters=0, n_samples=1000, funct=x_sinx, name_estimator="GradientBoostingRegressor", symmetry=True):
    """
    Checking what happens when you vary alpha for different types of splits of the dataset.
    """
    mapie_coverage = []
    interval_width = []
    list_scores = []

    if iters>0:
        for i in range(iters):
            mc, iw, ls = CQR_alphas(alphas, random_data=random_data, random_split=random_split, n_samples=n_samples, funct=funct, name_estimator=name_estimator, symmetry=symmetry)
            mapie_coverage.append(mc)
            interval_width.append(iw)
            list_scores.append(ls)
    else:
        mapie_coverage_ = []
        interval_width_ = []
        list_scores_ = []
        random_state = np.random.randint(0, 1000)
        X_train_, X_calib_, y_train_, y_calib_, X_test, y_test, _ = get_data(funct, n_samples=n_samples, random_state=random_state, random_data=random_data, random_split=random_split, data_name="paper_reproduction")
        for alpha in alphas:
            estimator = get_estimator(name_estimator)
            mapie_reg = MapieQuantileRegressor(
                estimator=estimator,
                alpha=alpha,
            )
            mapie_reg.fit(X_train_, y_train_, X_calib=X_calib_, y_calib=y_calib_)
            y_pred, y_pis, = mapie_reg.predict(X_test, symmetry=symmetry)

            mapie_coverage_.append(regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]))
            list_scores_.append(mapie_reg.conformity_scores_)
            interval_width_.append(regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0]))
        return mapie_coverage_, interval_width_, list_scores_
    return mapie_coverage, interval_width, list_scores