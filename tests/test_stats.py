import numpy as np

from deeplearners_toolkit.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    histogram,
    quantile
)

def test_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert mean(data) == 3.0

def test_median():
    data = np.array([1, 2, 3, 4, 5])
    assert median(data) == 3.0

def test_std():
    data = np.array([1, 2, 3, 4, 5])
    assert np.isclose(std(data), np.std(data))

def test_minimum_and_maximum():
    data = np.array([5, 2, 8, 1, 9])
    assert minimum(data) == 1
    assert maximum(data) == 9

def test_nan_handling():
    data = np.array([1, 2, np.nan, 4])
    assert np.isclose(mean(data),7/3)
    assert median(data) == 2.0
    assert minimum(data) == 1
    assert maximum(data) == 4

def test_axis_mean():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = mean(data, axis=0)
    expected = np.array([2.5, 3.5, 4.5])
    assert np.allclose(result, expected)

def test_axis_statistics():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(
        mean(data, axis=1),
        np.array([2, 5])
    )

    assert np.allclose(
        minimum(data, axis=0),
        np.array([1, 2, 3])
    )

    assert np.allclose(
        maximum(data, axis=0),
        np.array([4, 5, 6])
    )

def test_histogram():
    data = np.array([1, 2, 3, 4, 5])

    counts, edges = histogram(data, bins=2)

    assert counts.sum() == 5
    assert len(edges) == 3


def test_histogram_ignores_nan():
    data = np.array([1, 2, np.nan, 3])

    counts, edges = histogram(data, bins=2)

    assert counts.sum() == 3


def test_quantile():
    data = np.array([1, 2, 3, 4, 5])

    assert quantile(data, 0.5) == 3


def test_quantile_with_nan():
    data = np.array([1, 2, np.nan, 4, 5])

    assert quantile(data, 0.5) == 3