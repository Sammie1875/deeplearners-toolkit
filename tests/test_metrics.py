import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1,
    mse
)


def test_accuracy():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    assert accuracy(y_true, y_pred) == 0.75


def test_perfect_accuracy():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]

    assert accuracy(y_true, y_pred) == 1.0


def test_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    result = confusion_matrix(y_true, y_pred)

    expected = np.array([
        [1, 1],
        [1, 1]
    ])

    assert np.array_equal(result, expected)


def test_precision():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]

    assert np.isclose(precision(y_true, y_pred), 2 / 3)


def test_recall():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]

    assert np.isclose(recall(y_true, y_pred), 2 / 3)


def test_f1():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]

    assert np.isclose(f1(y_true, y_pred), 2 / 3)


def test_precision_no_positive_predictions():
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 0, 0]

    assert precision(y_true, y_pred) == 0.0


def test_recall_no_actual_positives():
    y_true = [0, 0, 0]
    y_pred = [0, 1, 0]

    assert recall(y_true, y_pred) == 0.0


def test_mse():
    y_true = [1, 2, 3]
    y_pred = [1, 3, 2]

    assert np.isclose(mse(y_true, y_pred), 2 / 3)


def test_mse_perfect_prediction():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]

    assert mse(y_true, y_pred) == 0.0


def test_different_shapes():
    y_true = [1, 2, 3]
    y_pred = [1, 2]

    with pytest.raises(ValueError):
        mse(y_true, y_pred)


def test_empty_accuracy():
    with pytest.raises(ValueError):
        accuracy([], [])