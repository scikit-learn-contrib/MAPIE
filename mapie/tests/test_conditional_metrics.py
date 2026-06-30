"""
Testing for conditional metrics module.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from mapie.metrics.conditional import (
    coverage_gap,
    excess_risk_target_coverage,
    worst_slab_coverage,
)


@pytest.mark.parametrize(
    "y, groups, confidence_level, y_intervals, weighted, expected_gap",
    [
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 1, 1]),
            0.75,
            np.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [1.0, 2.0],
                    [4.0, 5.0],
                ]
            )[:, :, np.newaxis],
            False,
            0.25,
        ),
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0, 0, 0, 1, 1, 2]),
            2 / 3,
            np.array(
                [
                    [-1.0, 0.0],
                    [0.0, 0.5],
                    [2.0, 2.0],
                    [3.1, 4.0],
                    [4.0, 5.0],
                    [5.0, 5.0],
                ]
            ),
            False,
            1 / 6,
        ),
        (
            np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0, 0, 1, 1, 1, 1]),
            0.5,
            np.array(
                [
                    [-2.0, -1.0],
                    [-0.5, 0.5],
                    [1.5, 2.5],
                    [2.0, 2.0],
                    [4.0, 5.0],
                    [3.5, 4.5],
                ]
            )[:, :, np.newaxis],
            True,
            1 / 6,
        ),
    ],
)
def test_coverage_gap_with_intervals(
    y: NDArray,
    groups: NDArray,
    confidence_level: float,
    y_intervals: NDArray,
    weighted: bool,
    expected_gap: float,
) -> None:
    """Test coverage gap with regression intervals."""
    gap = coverage_gap(
        y,
        groups,
        confidence_level,
        y_intervals=y_intervals,
        weighted=weighted,
    )
    np.testing.assert_allclose(gap, expected_gap)


@pytest.mark.parametrize(
    "y, groups, confidence_level, y_sets, weighted, expected_gap",
    [
        (
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
            0.75,
            np.array(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, False],
                ]
            )[:, :, np.newaxis],
            False,
            0.25,
        ),
        (
            np.array([0, 1, 2, 2, 3, 1]),
            np.array([0, 0, 0, 1, 1, 2]),
            2 / 3,
            np.array(
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, True, True],
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, True, True, False],
                ]
            ),
            False,
            1 / 6,
        ),
        (
            np.array([0, 1, 2, 2, 3, 1]),
            np.array([0, 0, 1, 1, 1, 1]),
            0.5,
            np.array(
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                    [False, False, True, True],
                    [False, False, True, True],
                    [False, True, True, False],
                ]
            )[:, :, np.newaxis],
            True,
            1 / 6,
        ),
    ],
)
def test_coverage_gap_with_sets(
    y: NDArray,
    groups: NDArray,
    confidence_level: float,
    y_sets: NDArray,
    weighted: bool,
    expected_gap: float,
) -> None:
    """Test coverage gap with classification prediction sets."""
    gap = coverage_gap(
        y,
        groups,
        confidence_level,
        y_sets=y_sets,
        weighted=weighted,
    )
    np.testing.assert_allclose(gap, expected_gap)


def test_coverage_gap_without_coverage_input() -> None:
    """Test coverage gap requires intervals or sets."""
    with pytest.raises(ValueError, match="Either y_intervals or y_sets"):
        coverage_gap(  # type: ignore[call-overload]
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1]), 0.75
        )


def test_coverage_gap_with_two_coverage_inputs() -> None:
    """Test coverage gap rejects ambiguous coverage inputs."""
    with pytest.raises(ValueError, match="Only one of y_intervals or y_sets"):
        coverage_gap(  # type: ignore[call-overload]
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 1, 1]),
            0.75,
            y_intervals=np.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [1.0, 2.0],
                    [4.0, 5.0],
                ]
            ),
            y_sets=np.array(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, False],
                ]
            ),
        )


def test_coverage_gap_with_multiple_interval_confidence_levels() -> None:
    """Test coverage gap rejects multiple interval confidence levels."""
    with pytest.raises(ValueError, match="only one confidence level"):
        coverage_gap(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 1, 1]),
            0.75,
            y_intervals=np.repeat(
                np.array(
                    [
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [1.0, 2.0],
                        [4.0, 5.0],
                    ]
                )[:, :, np.newaxis],
                2,
                axis=2,
            ),
        )


def test_coverage_gap_with_multiple_set_confidence_levels() -> None:
    """Test coverage gap rejects multiple set confidence levels."""
    with pytest.raises(ValueError, match="only one confidence level"):
        coverage_gap(
            np.array([0, 1, 0, 1]),
            np.array([0, 0, 1, 1]),
            0.75,
            y_sets=np.repeat(
                np.array(
                    [
                        [True, False],
                        [False, True],
                        [True, False],
                        [False, False],
                    ]
                )[:, :, np.newaxis],
                2,
                axis=2,
            ),
        )


def test_coverage_gap_with_float_class_labels() -> None:
    """Test coverage gap rejects non-integer class labels."""
    with pytest.raises(ValueError, match="integer class labels"):
        coverage_gap(
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0, 0, 1, 1]),
            0.75,
            y_sets=np.array(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, False],
                ]
            ),
        )


def test_coverage_gap_with_class_labels_outside_sets() -> None:
    """Test coverage gap rejects class labels outside prediction sets."""
    with pytest.raises(ValueError, match="outside y_sets"):
        coverage_gap(
            np.array([0, 1, 0, 2]),
            np.array([0, 0, 1, 1]),
            0.75,
            y_sets=np.array(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, False],
                ]
            ),
        )


def test_coverage_gap_with_float_groups() -> None:
    """Test coverage gap rejects non-integer groups."""
    with pytest.raises(ValueError, match="integer group labels"):
        coverage_gap(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            0.75,
            y_intervals=np.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                    [1.0, 2.0],
                    [4.0, 5.0],
                ]
            ),
        )


class DummyWSC:
    """Dummy covmetrics WSC backend for deterministic wrapper tests."""

    x: list[NDArray] = []
    cover: list[NDArray] = []
    params: list[tuple[float, int, int]] = []

    def evaluate(
        self,
        x: NDArray,
        cover: NDArray,
        delta: float,
        M: int,
        seed: int,
    ) -> float:
        """Store call arguments and return a fixed score."""
        self.x.append(x)
        self.cover.append(cover)
        self.params.append((delta, M, seed))
        return 0.5


@pytest.fixture
def dummy_wsc(monkeypatch: pytest.MonkeyPatch) -> type[DummyWSC]:
    """Patch covmetrics WSC with a deterministic dummy."""
    DummyWSC.x = []
    DummyWSC.cover = []
    DummyWSC.params = []
    monkeypatch.setattr("mapie.metrics.conditional.WSC", DummyWSC)
    return DummyWSC


def test_worst_slab_coverage_with_intervals(
    dummy_wsc: type[DummyWSC],
) -> None:
    """Test WSC with regression intervals."""
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    y_intervals = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [4.0, 5.0]])

    score = worst_slab_coverage(
        x,
        y,
        y_intervals=y_intervals,
        delta=0.5,
        n_directions=7,
        random_state=123,
    )

    assert score == 0.5
    np.testing.assert_array_equal(dummy_wsc.x[0], x)
    np.testing.assert_array_equal(dummy_wsc.cover[0], np.array([1, 1, 1, 0]))
    assert dummy_wsc.params == [(0.5, 7, 123)]


def test_worst_slab_coverage_with_sets(dummy_wsc: type[DummyWSC]) -> None:
    """Test WSC with classification prediction sets."""
    x = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    y_sets = np.array(
        [
            [True, False],
            [False, True],
            [False, True],
            [True, True],
        ]
    )

    score = worst_slab_coverage(
        x,
        y,
        y_sets=y_sets,
        delta=0.25,
        n_directions=3,
        random_state=321,
    )

    assert score == 0.5
    np.testing.assert_array_equal(dummy_wsc.cover[0], np.array([1, 1, 0, 1]))
    assert dummy_wsc.params == [(0.25, 3, 321)]


def test_worst_slab_coverage_without_coverage_input() -> None:
    """Test WSC requires intervals or sets."""
    with pytest.raises(ValueError, match="Either y_intervals or y_sets"):
        worst_slab_coverage(  # type: ignore[call-overload]
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0])
        )


def test_worst_slab_coverage_with_two_coverage_inputs() -> None:
    """Test WSC rejects ambiguous coverage inputs."""
    with pytest.raises(ValueError, match="Only one of y_intervals or y_sets"):
        worst_slab_coverage(  # type: ignore[call-overload]
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
            y_sets=np.array([[True, False], [False, True]]),
        )


@pytest.mark.parametrize(
    "x, delta, n_directions, match",
    [
        (np.array([0.0, 1.0]), 0.5, 10, "2D array"),
        (np.array([[0.0], [np.nan]]), 0.5, 10, "NaN"),
        (np.array([[0.0], [1.0]]), "0.5", 10, "delta"),
        (np.array([[0.0], [1.0]]), 0.0, 10, "delta"),
        (np.array([[0.0], [1.0]]), 1.0, 10, "delta"),
        (np.array([[0.0], [1.0]]), 0.5, 1.5, "n_directions"),
        (np.array([[0.0], [1.0]]), 0.5, 0, "n_directions"),
    ],
)
def test_worst_slab_coverage_invalid_inputs(
    x: NDArray,
    delta: float,
    n_directions: int,
    match: str,
) -> None:
    """Test WSC input validation."""
    with pytest.raises(ValueError, match=match):
        worst_slab_coverage(
            x,
            np.array([0.0, 1.0]),
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
            delta=delta,
            n_directions=n_directions,
        )


def test_worst_slab_coverage_rejects_length_mismatch() -> None:
    """Test WSC rejects inconsistent x and coverage lengths."""
    with pytest.raises(ValueError, match="different length"):
        worst_slab_coverage(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0.0, 1.0]),
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )


class DummyERT:
    """Dummy covmetrics ERT backend for deterministic wrapper tests."""

    x: list[NDArray] = []
    cover: list[NDArray] = []
    params: list[tuple[float, int, int, object]] = []
    model_cls: list[object] = []
    model_kwargs: list[dict[str, object]] = []
    fit_kwargs: list[dict[str, object]] = []

    def __init__(self, model_cls: object = None, **model_kwargs: object) -> None:
        self.model_cls.append(model_cls)
        self.model_kwargs.append(model_kwargs)

    def evaluate(
        self,
        x: NDArray,
        cover: NDArray,
        alpha: float,
        n_splits: int,
        random_state: int,
        loss: object,
        **fit_kwargs: object,
    ) -> float:
        """Store call arguments and return a fixed score."""
        self.x.append(x)
        self.cover.append(cover)
        self.params.append((alpha, n_splits, random_state, loss))
        self.fit_kwargs.append(fit_kwargs)
        return -0.25


@pytest.fixture
def dummy_ert(monkeypatch: pytest.MonkeyPatch) -> type[DummyERT]:
    """Patch covmetrics ERT with a deterministic dummy."""
    DummyERT.x = []
    DummyERT.cover = []
    DummyERT.params = []
    DummyERT.model_cls = []
    DummyERT.model_kwargs = []
    DummyERT.fit_kwargs = []
    monkeypatch.setattr("mapie.metrics.conditional.ERT", DummyERT)
    return DummyERT


def test_excess_risk_target_coverage_with_intervals(
    dummy_ert: type[DummyERT],
) -> None:
    """Test ERT with regression intervals."""
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    y_intervals = np.array([[0.0, 1.0], [0.0, 2.0], [3.0, 4.0], [4.0, 5.0]])
    loss = object()

    score = excess_risk_target_coverage(
        x,
        y,
        0.75,
        y_intervals=y_intervals,
        model_kwargs={"max_iter": 3},
        n_splits=2,
        random_state=123,
        loss=loss,
        fit_kwargs={"sample_weight": np.ones(2)},
    )

    assert score == -0.25
    np.testing.assert_array_equal(dummy_ert.x[0], x)
    np.testing.assert_array_equal(dummy_ert.cover[0], np.array([1, 1, 0, 0]))
    assert dummy_ert.params == [(0.25, 2, 123, loss)]
    assert dummy_ert.model_kwargs == [{"max_iter": 3, "random_state": 123}]
    np.testing.assert_array_equal(dummy_ert.fit_kwargs[0]["sample_weight"], np.ones(2))


def test_excess_risk_target_coverage_with_sets(dummy_ert: type[DummyERT]) -> None:
    """Test ERT with classification prediction sets."""
    x = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    y_sets = np.array(
        [
            [True, False],
            [False, True],
            [False, True],
            [True, True],
        ]
    )

    score = excess_risk_target_coverage(
        x,
        y,
        0.8,
        y_sets=y_sets,
        model_cls=dict,
        model_kwargs={"custom": True},
        n_splits=3,
        random_state=321,
    )

    assert score == -0.25
    np.testing.assert_array_equal(dummy_ert.cover[0], np.array([1, 1, 0, 1]))
    assert dummy_ert.params == [(0.2, 3, 321, None)]
    assert dummy_ert.model_cls == [dict]
    assert dummy_ert.model_kwargs == [{"custom": True}]


def test_excess_risk_target_coverage_without_coverage_input() -> None:
    """Test ERT requires intervals or sets."""
    with pytest.raises(ValueError, match="Either y_intervals or y_sets"):
        excess_risk_target_coverage(  # type: ignore[call-overload]
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), 0.75
        )


def test_excess_risk_target_coverage_with_two_coverage_inputs() -> None:
    """Test ERT rejects ambiguous coverage inputs."""
    with pytest.raises(ValueError, match="Only one of y_intervals or y_sets"):
        excess_risk_target_coverage(  # type: ignore[call-overload]
            np.array([[0.0], [1.0]]),
            np.array([0, 1]),
            0.75,
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
            y_sets=np.array([[True, False], [False, True]]),
        )


@pytest.mark.parametrize(
    "x, n_splits, match",
    [
        (np.array([0.0, 1.0]), 2, "2D array"),
        (np.array([[0.0], [np.nan]]), 2, "NaN"),
        (np.array([[0.0], [1.0]]), 1, "n_splits"),
        (np.array([[0.0], [1.0]]), 2.5, "n_splits"),
    ],
)
def test_excess_risk_target_coverage_invalid_inputs(
    x: NDArray,
    n_splits: int,
    match: str,
) -> None:
    """Test ERT input validation."""
    with pytest.raises(ValueError, match=match):
        excess_risk_target_coverage(
            x,
            np.array([0.0, 1.0]),
            0.75,
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
            n_splits=n_splits,
        )


def test_excess_risk_target_coverage_rejects_length_mismatch() -> None:
    """Test ERT rejects inconsistent x and coverage lengths."""
    with pytest.raises(ValueError, match="different length"):
        excess_risk_target_coverage(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0.0, 1.0]),
            0.75,
            y_intervals=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
