"""
Testing for conditional metrics module.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from mapie.metrics.conditional import coverage_gap


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
