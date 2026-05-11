class ExchangeabilityTestAdrien:
    """
    Examples
    --------

    # run before splitting test data into calibration and test
    exchangeability_test = ExchangeabilityTestAdrien()
    exchangeability_test.run(X_test, y_test)

    # after deployment
    while True:
        x_t = get_data()
        y_pred = model.predict(x_t)
        y_t = get_true_label()
        exchangeability_test.update(x_t, y_t, y_pred)
    """

    def __init__(self, method="abc", threshold=None, score=None) -> None:
        pass

    def run(X, y, y_pred=None) -> bool:
        """
        For a fixed dataset, run once the algo.
        Some methods can require y_pred (e.g. when using a conformity score).
        """
        is_exchangeable = ...
        return is_exchangeable

    def update(X, y, y_pred=None) -> bool:
        """
        For sequential data, update result with one or more data points.
        Some methods can require y_pred (e.g. when using a conformity score).
        """
        is_exchangeable = ...
        return is_exchangeable

    def summary() -> None:
        """
        Print some statistics, plot some curves.
        """
