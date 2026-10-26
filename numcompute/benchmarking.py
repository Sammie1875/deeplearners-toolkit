
import time
import numpy as np

from numcompute.stats import mean
from numcompute.metrics import mse
from numcompute.preprocessing import StandardScaler


# Python loop versions used only for benchmarking

def loop_mean(data):
    total = 0.0

    for value in data:
        total += value

    return total / len(data)


def loop_mse(y_true, y_pred):
    total_error = 0.0

    for actual, predicted in zip(y_true, y_pred):
        difference = actual - predicted
        total_error += difference ** 2

    return total_error / len(y_true)


def loop_standard_scaler(data):
    """
    Standardise each column using Python loops.
    Uses population standard deviation (ddof=0).
    """
    rows, cols = data.shape
    result = np.zeros_like(data, dtype=float)

    for col in range(cols):
        total = 0.0

        for row in range(rows):
            total += data[row, col]

        column_mean = total / rows

        variance_sum = 0.0

        for row in range(rows):
            variance_sum += (data[row, col] - column_mean) ** 2

        column_std = (variance_sum / rows) ** 0.5

        if column_std == 0:
            column_std = 1.0

        for row in range(rows):
            result[row, col] = (
                data[row, col] - column_mean
            ) / column_std

    return result


# Reusable timing function

def measure_time(function, repeats=5):
    """
    Run a function repeatedly and return the median
    execution time in seconds.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    times = []

    # Warm-up run
    function()

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        end = time.perf_counter()

        times.append(end - start)

    return float(np.median(times))


# Benchmark 1: Mean

def benchmark_mean(size=100000, repeats=5):
    rng = np.random.default_rng(42)
    data = rng.random(size)

    if size < 1:
        raise ValueError("size must be positive")

    expected = loop_mean(data)
    actual = mean(data)

    if not np.isclose(expected, actual):
        raise AssertionError("Mean implementations disagree")

    loop_time = measure_time(
        lambda: loop_mean(data), repeats
    )

    vector_time = measure_time(
        lambda: mean(data), repeats
    )

    return {
        "operation": "Mean",
        "size": size,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Benchmark 2: Mean Squared Error

def benchmark_mse(size=100000, repeats=5):
    if size < 1:
        raise ValueError("size must be positive")

    rng = np.random.default_rng(42)

    y_true = rng.random(size)
    y_pred = rng.random(size)

    expected = loop_mse(y_true, y_pred)
    actual = mse(y_true, y_pred)

    if not np.isclose(expected, actual):
        raise AssertionError("MSE implementations disagree")

    loop_time = measure_time(
        lambda: loop_mse(y_true, y_pred), repeats
    )

    vector_time = measure_time(
        lambda: mse(y_true, y_pred), repeats
    )

    return {
        "operation": "MSE",
        "size": size,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Benchmark 3: StandardScaler

def benchmark_scaler(rows=10000, cols=5, repeats=5):
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")

    rng = np.random.default_rng(42)
    data = rng.random((rows, cols))

    def vector_scaler():
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    expected = loop_standard_scaler(data)
    actual = vector_scaler()

    if not np.allclose(expected, actual, atol=1e-8):
        raise AssertionError(
            "StandardScaler implementations disagree"
        )

    loop_time = measure_time(
        lambda: loop_standard_scaler(data), repeats
    )

    vector_time = measure_time(
        vector_scaler, repeats
    )

    return {
        "operation": "StandardScaler",
        "size": rows * cols,
        "loop_time": loop_time,
        "vector_time": vector_time,
        "speedup": loop_time / vector_time
    }


# Display all benchmark results

def run_benchmarks():
    print("\nNUMCOMPUTE - PERFORMANCE BENCHMARKS")
    print("=" * 82)

    print(
        f"{'Operation':<18}"
        f"{'Elements':>12}"
        f"{'Loop (ms)':>16}"
        f"{'NumPy (ms)':>16}"
        f"{'Speedup':>14}"
    )

    print("-" * 82)

    results = [
        benchmark_mean(),
        benchmark_mse(),
        benchmark_scaler()
    ]

    for result in results:
        loop_ms = result["loop_time"] * 1000
        vector_ms = result["vector_time"] * 1000

        print(
            f"{result['operation']:<18}"
            f"{result['size']:>12,}"
            f"{loop_ms:>16.4f}"
            f"{vector_ms:>16.4f}"
            f"{result['speedup']:>13.2f}x"
        )

    print("=" * 82)
    print("Random seed: 42")
    print("Timing: median of 5 runs, after one warm-up")
    print("All correctness checks passed.")

    return results


if __name__ == "__main__":
    run_benchmarks()
