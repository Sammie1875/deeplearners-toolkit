
import numpy as np
import pytest

from deeplearners_toolkit.benchmarking import (
    loop_mean,
    loop_mse,
    loop_standard_scaler,
    measure_time,
    benchmark_mean,
    benchmark_mse,
    benchmark_scaler
)


def test_loop_mean():
    data = np.array([1, 2, 3, 4, 5])
    assert loop_mean(data) == 3.0


def test_loop_mse():
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 3, 2])

    assert np.isclose(loop_mse(actual, predicted), 2 / 3)


def test_loop_standard_scaler():
    data = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])

    result = loop_standard_scaler(data)

    assert np.allclose(np.mean(result, axis=0), [0, 0])
    assert np.allclose(np.std(result, axis=0), [1, 1])


def test_mean_benchmark():
    result = benchmark_mean(size=100, repeats=2)

    assert result["operation"] == "Mean"
    assert result["loop_time"] >= 0
    assert result["vector_time"] > 0


def test_mse_benchmark():
    result = benchmark_mse(size=100, repeats=2)

    assert result["operation"] == "MSE"
    assert result["speedup"] > 0


def test_scaler_benchmark():
    result = benchmark_scaler(rows=20, cols=3, repeats=2)

    assert result["operation"] == "StandardScaler"
    assert result["speedup"] > 0


def test_invalid_repeats():
    with pytest.raises(ValueError):
        measure_time(lambda: None, repeats=0)
