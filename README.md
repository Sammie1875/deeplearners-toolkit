# NumCompute

**A lightweight numerical computing toolkit built with Python and NumPy.**

NumCompute provides utilities for numerical differentiation, sorting and searching, statistical analysis, ranking, classification and regression metrics, data preprocessing, CSV loading, and pipeline composition.

The package is designed to offer useful numerical and data-processing operations through a simple API, without relying on libraries such as SciPy, pandas, or scikit-learn.

## Features

- **Sorting and searching:** Stable sorting, multi-key sorting, binary search, top-\(k\) selection, and quick selection.
- **Numerical differentiation:** Finite-difference gradient and Jacobian estimation, including batched variants.
- **Statistics:** Mean, median, standard deviation, minimum, maximum, histogram, and quantile calculations.
- **Ranking:** Multiple ranking methods, configurable ordering, axis support, and NaN handling.
- **Evaluation metrics:** Accuracy, confusion matrix, precision, recall, F1 score, and mean squared error.
- **Preprocessing:** Standard scaling, min-max scaling, and one-hot encoding.
- **Pipelines:** Sequential preprocessing and optional final estimator integration.
- **Input/output:** CSV loading into NumPy arrays.

## Installation

### Prerequisites

- Python 3.10 or later
- Git
- NumPy

### Install directly from Git

```bash
python -m pip install "https://github.com/Sammie1875/deeplearners-toolkit.git"
```


### Install from a local clone

```bash
git clone https://github.com/Sammie1875/deeplearners-toolkit.git
cd deeplearners-toolkit
python -m pip install .
```

For development, install in editable mode:

```bash
python -m pip install -e .
```

## Quick start

### Sorting and searching

```python
import numpy as np
from numcompute import stable_sort, binary_search

values = np.array([5, 2, 8, 1, 3])

sorted_values = stable_sort(values)
position, found = binary_search(sorted_values, 3)

print(sorted_values)
# [1 2 3 5 8]

print(position, found)
# 2 True
```

### Statistical calculations

```python
import numpy as np
from numcompute import mean, median, std

values = np.array([10, 20, 30, 40, 50])

print(mean(values))    # 30.0
print(median(values))  # 30.0
print(std(values))     # Population standard deviation
```

### Numerical differentiation

```python
import numpy as np
from numcompute import gradient

def f(x):
    return np.sum(x ** 2)

x = np.array([2.0, 3.0])

g = gradient(f, x, method="central")
print(g)
# Approximately [4. 6.]
```

### Classification metrics

```python
import numpy as np
from numcompute import accuracy, precision, recall, f1

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0])

print(accuracy(y_true, y_pred))  # 0.8
print(precision(y_true, y_pred)) # 1.0
print(recall(y_true, y_pred))    # 2/3
print(f1(y_true, y_pred))        # 0.8
```

### Data preprocessing

```python
import numpy as np
from numcompute import StandardScaler

X = np.array([
    [10.0, 100.0],
    [20.0, 200.0],
    [30.0, 300.0],
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

### Pipeline composition

```python
from numcompute import Pipeline, StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
])

X_transformed = pipe.fit_transform(X)
```

The pipeline supports preprocessing components that implement `fit()` and `transform()`. Its optional estimator support depends on the estimator implementing the expected `fit()` and `predict()` interface.

## API overview

### 1. `io.py` — Input/output

| Function | Description |
|---|---|
| `read_csv(path, delimiter=",", missing_value=np.nan, dtype=float)` | Loads CSV data into a NumPy array using NumPy's `genfromtxt`. |

### 2. `sort_search.py` — Sorting and searching

| Function | Description |
|---|---|
| `stable_sort(values, axis=-1)` | Sorts array elements along a specified axis. |
| `multi_key_sort(data, ascending=True)` | Returns indices that order rows using multiple keys. |
| `binary_search(sorted_array, x)` | Returns the insertion index and whether the target exists. |
| `top_k(values, k, largest=True, return_indices=True)` | Selects the \(k\) largest or smallest elements or their indices. |
| `quick_select(values, k, largest=True, return_index=True)` | Selects an element by rank without fully sorting the input. |

Check each function's docstring for index conventions and output ordering.

### 3. `optim.py` — Numerical differentiation

| Function | Description |
|---|---|
| `gradient(f, x, h=1e-5, method="central")` | Estimates the gradient of a scalar-valued function. |
| `jacobian(F, x, h=1e-5, method="central")` | Estimates the Jacobian of a vector-valued function. |
| `gradient_batched(f, x, h=1e-5, method="central")` | Estimates gradients using batched function evaluations. |
| `jacobian_batched(F, x, h=1e-5, method="central")` | Estimates Jacobians using batched function evaluations. |

Supported finite-difference methods are central and forward differences.

### 4. `stats.py` — Descriptive statistics

| Function | Description |
|---|---|
| `mean(data, axis=None, ignore_nan=True)` | Calculates arithmetic means. |
| `median(data, axis=None, ignore_nan=True)` | Calculates medians. |
| `std(data, axis=None, ignore_nan=True, ddof=0)` | Calculates standard deviations. |
| `minimum(data, axis=None, ignore_nan=True)` | Calculates minimum values. |
| `maximum(data, axis=None, ignore_nan=True)` | Calculates maximum values. |
| `histogram(data, bins=10, range=None, ignore_nan=True)` | Computes histogram counts and bin edges. |
| `quantile(data, q, axis=None, ignore_nan=True)` | Computes quantiles. |

### 5. `rank.py` — Ranking and percentile utilities

| Function | Description |
|---|---|
| `rankdata(values, method="average", ascending=True, axis=None, nan_policy="raise")` | Computes observation ranks. |
| `percentile_rank(values, score, kind="rank", nan_policy="raise")` | Calculates the percentile rank of one or more scores. |

Supported ranking methods are `average`, `dense`, `ordinal`, `min`, and `max`. NaN policies are `raise`, `omit`, and `propagate`.

Supported percentile-rank kinds are `rank`, `weak`, `strict`, and `mean`.

### 6. `metrics.py` — Evaluation metrics

| Function | Description |
|---|---|
| `accuracy(y_true, y_pred)` | Calculates the proportion of correct predictions. |
| `confusion_matrix(y_true, y_pred, labels=None)` | Counts true versus predicted class combinations. |
| `precision(y_true, y_pred, positive_label=1)` | Calculates positive predictive value. |
| `recall(y_true, y_pred, positive_label=1)` | Calculates the proportion of actual positives identified. |
| `f1(y_true, y_pred, positive_label=1)` | Calculates the harmonic mean of precision and recall. |
| `mse(y_true, y_pred)` | Calculates mean squared error for numerical predictions. |

Precision, recall, and F1 use a specified positive class and are intended for binary classification.

### 7. `preprocessing.py` — Data preprocessing

| Class | Methods | Description |
|---|---|---|
| `StandardScaler` | `fit`, `transform`, `fit_transform` | Centers features and scales by standard deviation. |
| `MinMaxScaler` | `fit`, `transform`, `fit_transform` | Scales features to a specified range. |
| `OneHotEncoder` | `fit`, `transform`, `fit_transform` | Encodes categorical columns as binary features. |

Fit preprocessing components on training data and reuse the fitted components to transform new data.

### 8. `pipeline.py` — Pipeline composition

| Class or method | Description |
|---|---|
| `Pipeline(steps)` | Creates an ordered sequence of named components. |
| `fit_transform(X)` | Fits preprocessing steps and returns transformed data. |
| `fit(X, y)` | Fits preprocessing steps and an optional final estimator. |
| `transform(X)` | Applies fitted preprocessing steps. |
| `predict(X)` | Transforms input and predicts using the final estimator. |

## Performance and complexity

The following table gives theoretical or typical complexity estimates, rather than benchmark measurements. Actual performance depends on array dimensions, data types, NumPy's implementation, and the cost of user-supplied functions.

Let:

- \(n\) = number of input elements or samples.
- \(d\) = number of features or input dimensions.
- \(c\) = number of classes.
- \(m\) = number of output dimensions.
- \(b\) = number of histogram bins.
- \(q\) = number of queried scores for percentile ranking.

### Sorting and searching

| Operation | Time complexity | Auxiliary space |
|---|---:|---:|
| `stable_sort()` | \(O(n\log n)\) typical | \(O(n)\) typical |
| `multi_key_sort()` | \(O(n\log n)\) | \(O(n)\) |
| `binary_search()` | \(O(\log n)\) | \(O(1)\) |
| `top_k()` | Typically \(O(n)\) | \(O(n)\) if indices are returned; otherwise implementation-dependent |
| `quick_select()` | Typically \(O(n)\) | \(O(n)\) if indices are returned; otherwise implementation-dependent |

Selection operations using partitioning typically avoid the \(O(n\log n)\) cost of a complete sort. Their precise bounds depend on the underlying implementation.

### Numerical differentiation

Assuming each function evaluation has cost \(C_f\) for `gradient()` or \(C_F\) for `jacobian()`:

| Operation | Function evaluations | Time complexity | Output space |
|---|---:|---:|---:|
| `gradient()` with central differences | \(2d\), plus validation/baseline evaluations if applicable | \(O(dC_f)\) | \(O(d)\) |
| `gradient()` with forward differences | \(d+1\) | \(O(dC_f)\) | \(O(d)\) |
| `jacobian()` with central differences | \(2d\), plus any baseline evaluation | \(O(dC_F)\) | \(O(md)\) |
| `jacobian()` with forward differences | \(d+1\) | \(O(dC_F)\) | \(O(md)\) |

Batched variants may reduce Python-call overhead, but the cost of evaluating the function over the batch still depends on its implementation. Their temporary memory use can be larger because perturbations are constructed together.

### Statistics

For a one-dimensional input of \(n\) elements:

| Function | Typical time complexity | Auxiliary space |
|---|---:|---:|
| `mean()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `median()` | Typically \(O(n)\) using selection-based algorithms | \(O(n)\) worst case |
| `std()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `minimum()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `maximum()` | \(O(n)\) | \(O(n)\) worst case for conversions or temporaries |
| `histogram()` | Typically \(O(n+b)\) | \(O(n+b)\) worst case |
| `quantile()` | Typically selection-based; depends on requested quantiles and NumPy implementation | Depends on input conversion and temporary arrays |

Axis-based operations scale with the total number of input elements. NaN-aware operations may have additional overhead.

### Ranking

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `rankdata()` on \(n\) values | \(O(n\log n)\) | \(O(n)\) |
| `rankdata()` on multidimensional arrays | \(O(N\log k)\) typical, where \(N\) is total elements and \(k\) is the ranked axis length | \(O(N)\) |
| `percentile_rank()` | \(O(n\log n + q\log n)\) | \(O(n+q)\) |

`rankdata()` sorts each ranked slice. `percentile_rank()` sorts the reference data once and uses binary-search insertion positions for the queries.

### Evaluation metrics

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `accuracy()` | \(O(n)\) | \(O(n)\) |
| `confusion_matrix()` | \(O(nc^2+c^2)\) for the provided matrix-multiplication implementation | \(O(nc+c^2)\) |
| `precision()` | \(O(n)\) | \(O(n)\) |
| `recall()` | \(O(n)\) | \(O(n)\) |
| `f1()` | \(O(n)\) | \(O(n)\) |
| `mse()` | \(O(n)\) | \(O(n)\) |

The confusion-matrix implementation constructs Boolean match arrays and multiplies them to count class combinations. This is more expensive in time and temporary memory than a conventional single-pass counting implementation.

### Preprocessing

For \(n\) samples, \(d\) input features, and \(C\) total one-hot output features:

| Operation | Time complexity | Auxiliary space |
|---|---:|---:|
| `StandardScaler.fit()` | \(O(nd)\) | \(O(d)\) stored statistics |
| `StandardScaler.transform()` | \(O(nd)\) | \(O(nd)\) for the output |
| `MinMaxScaler.fit()` | \(O(nd)\) | \(O(d)\) stored statistics |
| `MinMaxScaler.transform()` | \(O(nd)\) | \(O(nd)\) for the output |
| `OneHotEncoder.fit()` | Typically \(O(nd\log n)\) | \(O(C)\) for stored categories |
| `OneHotEncoder.transform()` | \(O(nC)\) | \(O(nC)\) for the encoded output |

The exact complexity of one-hot encoding depends on the number of unique categories in each feature and the implementation's intermediate arrays.

### Input/output

| Function | Time complexity | Auxiliary space |
|---|---:|---:|
| `read_csv()` | \(O(N)\) typical for \(N\) parsed input values | \(O(N)\) for the resulting array |

CSV parsing also depends on file size, data types, delimiters, and missing-value handling.

### Pipeline

For a pipeline with \(s\) preprocessing steps, its complexity is the sum of the costs of its component operations.

| Operation | Time complexity | Auxiliary space |
|---|---|---|
| `fit_transform()` | Sum of component fit and transform costs | Depends on intermediate transformed arrays |
| `fit()` | Sum of preprocessing costs and final estimator fitting cost | Depends on component state and intermediate arrays |
| `transform()` | Sum of preprocessing transform costs | Depends on intermediate transformed arrays |
| `predict()` | Sum of preprocessing transform costs and estimator prediction cost | Depends on intermediate arrays and predictions |

These estimates describe algorithmic growth, not measured execution speed. Benchmark results should be reported separately with the machine, Python version, NumPy version, input sizes, and benchmark method.