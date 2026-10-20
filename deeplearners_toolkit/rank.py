
"""
rank.py
=======

Ranking and percentile utilities for the DeepLearners Toolkit.

Dependencies:
    Python standard library
    NumPy

Public functions:
    rankdata()
    percentile_rank()

Supported ranking methods:
    average  - Tied values receive the average rank.
    dense    - Tied values share a rank; no gaps between groups.
    ordinal  - Every observation receives a unique rank.
    min      - Tied values receive the smallest rank in their group.
    max      - Tied values receive the largest rank in their group.

Conventions:
    ascending=True  -> smallest value gets rank 1.
    ascending=False -> largest value gets rank 1.
    Ranks are mapped back to the original input order.

NaN policies:
    raise     -> reject NaN values.
    omit      -> exclude NaNs and leave NaN at their original positions.
    propagate -> if a ranking slice contains NaN, its ranks are all NaN.

This module does not use pandas, SciPy or scikit-learn.
"""

import numpy as np


# ============================================================
# 1. VALIDATION HELPERS
# ============================================================

_RANK_METHODS = {"average", "dense", "ordinal", "min", "max"}
_NAN_POLICIES = {"raise", "omit", "propagate"}
_PERCENTILE_METHODS = {"rank", "weak", "strict", "mean"}


def _as_real_array(values, name="values"):
    """
    Convert input to a NumPy array and validate its numeric type.

    Boolean, string, complex and nonnumeric arrays are rejected.
    Integer inputs remain integers to preserve sorting precision.
    """
    try:
        arr = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be convertible to a numeric array"
        ) from exc

    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")

    if np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(f"{name} must not contain complex values")

    return arr


def _validate_method(method):
    """Reject unsupported ranking method names."""
    if method not in _RANK_METHODS:
        choices = ", ".join(sorted(_RANK_METHODS))
        raise ValueError(
            f"Invalid method {method!r}. Choose from: {choices}"
        )


def _validate_nan_policy(nan_policy):
    """Validate the policy used when input contains NaNs."""
    if nan_policy not in _NAN_POLICIES:
        choices = ", ".join(sorted(_NAN_POLICIES))
        raise ValueError(
            f"Invalid nan_policy {nan_policy!r}. "
            f"Choose from: {choices}"
        )


def _validate_axis(axis, ndim):
    """Validate an axis and convert negative axes to positive ones."""
    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise TypeError("axis must be an integer or None")

    if ndim == 0:
        raise ValueError("axis cannot be specified for scalar input")

    axis = int(axis)

    if not -ndim <= axis < ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for {ndim} dimensions"
        )

    return axis % ndim


# ============================================================
# 2. CORE RANKING ALGORITHM
# ============================================================

def _rank_1d_valid(values, method, ascending):
    """
    Rank a one-dimensional array that contains no NaNs.

    This internal function assumes that method and ascending
    have already been validated.

    Stable sorting ensures that equal values retain their
    original relative order, including descending ranking.
    """
    n = len(values)
    ranks = np.empty(n, dtype=float)

    if n == 0:
        return ranks

    # Obtain indices in ascending order without changing the input.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    # Identify the start of every distinct-value group.
    # For example, [10, 10, 20, 30, 30] gives groups:
    # [10, 10], [20], [30, 30].
    starts = np.r_[
        0,
        np.flatnonzero(
            sorted_values[1:] != sorted_values[:-1]
        ) + 1,
    ]
    ends = np.r_[starts[1:], n]

    # Reverse whole groups for descending ranking, not individual
    # elements. This preserves original order among tied values.
    if not ascending:
        groups = [
            order[start:end]
            for start, end in zip(starts, ends)
        ]
        order = np.concatenate(groups[::-1])

        # Values are now grouped in descending order.
        sorted_values = values[order]

        starts = np.r_[
            0,
            np.flatnonzero(
                sorted_values[1:] != sorted_values[:-1]
            ) + 1,
        ]
        ends = np.r_[starts[1:], n]

    # Ordinal ranking assigns a unique position to every item.
    if method == "ordinal":
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks

    # Assign a rank to each group of equal values.
    for dense_rank, (start, end) in enumerate(
        zip(starts, ends), start=1
    ):
        first_position = start + 1  # Convert to one-based ranking.
        last_position = end

        if method == "average":
            rank = (first_position + last_position) / 2.0

        elif method == "dense":
            rank = float(dense_rank)

        elif method == "min":
            rank = float(first_position)

        else:  # method == "max"
            rank = float(last_position)

        ranks[order[start:end]] = rank

    return ranks


def _rank_1d(values, method, ascending, nan_policy):
    """
    Apply NaN handling before ranking a one-dimensional array.

    Kept separate from rankdata() so each axis can use the
    same implementation.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=float)

    if n == 0:
        return result

    nan_mask = np.isnan(values)

    if not np.any(nan_mask):
        return _rank_1d_valid(values, method, ascending)

    if nan_policy == "raise":
        raise ValueError(
            "Input contains NaN values. Use nan_policy='omit' "
            "or nan_policy='propagate' to choose another policy."
        )

    if nan_policy == "propagate":
        return result

    # Omit NaNs from ranking while retaining their original positions.
    valid_values = values[~nan_mask]
    result[~nan_mask] = _rank_1d_valid(
        valid_values, method, ascending
    )

    return result


# ============================================================
# 3. PUBLIC RANKING FUNCTION
# ============================================================

def rankdata(
    values,
    method="average",
    ascending=True,
    axis=None,
    nan_policy="raise",
):
    """
    Calculate ranks while preserving the original input order.

    Parameters
    ----------
    values : array-like
        Real numeric values to rank.

    method : str, default="average"
        "average", "dense", "ordinal", "min", or "max".

    ascending : bool, default=True
        True means the smallest value gets rank 1.
        False means the largest value gets rank 1.

    axis : int or None, default=None
        None flattens the array before ranking.
        An integer ranks independently along that axis.

    nan_policy : str, default="raise"
        "raise": reject NaNs.
        "omit": leave NaN ranks at missing positions.
        "propagate": return all NaN ranks for an affected slice.

    Returns
    -------
    np.ndarray
        Floating-point ranks in the original input shape when
        axis is specified, or a flattened array when axis=None.

    Notes
    -----
    Infinities are permitted and ranked numerically.
    Ordinal ties preserve original observation order.

    Examples
    --------
    >>> rankdata([40, 10, 30, 10, 50])
    array([4. , 1.5, 3. , 1.5, 5. ])

    >>> rankdata([40, 10, 30, 10, 50], method="dense")
    array([3., 1., 2., 1., 4.])

    >>> rankdata([10, 20, 20, 40], ascending=False)
    array([4. , 2.5, 2.5, 1. ])

    >>> rankdata([[30, 10], [20, 40]], axis=1)
    array([[2., 1.], [1., 2.]])
    """
    arr = _as_real_array(values)
    _validate_method(method)
    _validate_nan_policy(nan_policy)

    if not isinstance(ascending, (bool, np.bool_)):
        raise TypeError("ascending must be a Boolean")

    # Convert only floating-point arrays to float. Keep integers
    # in their original dtype so large integer values remain distinct.
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float, copy=False)

    if axis is None:
        return _rank_1d(
            arr.ravel(), method, bool(ascending), nan_policy
        )

    axis = _validate_axis(axis, arr.ndim)

    # Move the selected axis to the end so each slice is handled
    # by the same one-dimensional ranking implementation.
    moved = np.moveaxis(arr, axis, -1)
    result = np.empty(moved.shape, dtype=float)

    for index in np.ndindex(moved.shape[:-1]):
        result[index] = _rank_1d(
            moved[index], method, bool(ascending), nan_policy
        )

    # Restore the original axis order and shape.
    return np.moveaxis(result, -1, axis)


# ============================================================
# 4. PERCENTILE-RANK FUNCTION
# ============================================================

def percentile_rank(
    values,
    score,
    kind="rank",
    nan_policy="raise",
):
    """
    Calculate the percentile rank of one or more scores.

    The percentile rank describes relative standing. It is not
    the same as calculating a percentile threshold.

    Parameters
    ----------
    values : array-like
        One-dimensional reference dataset.

    score : scalar or array-like
        Score or scores to evaluate.

    kind : str, default="rank"
        "weak":
            100 * count(values <= score) / n

        "strict":
            100 * count(values < score) / n

        "mean":
            Average of the weak and strict percentages.

        "rank":
            Average percentage rank of matching observations.
            If a score is absent, use its strict percentile rank.

    nan_policy : str, default="raise"
        "raise": reject NaNs in the reference data.
        "omit": exclude NaNs from the reference data.
        "propagate": return NaN results if reference data contain NaN.

    Returns
    -------
    float or np.ndarray
        A float for scalar input, otherwise an array matching
        the shape of score.

    Raises
    ------
    ValueError
        If the reference dataset is empty, contains NaNs under
        the "raise" policy, or becomes empty after omission.
    """
    arr = _as_real_array(values, name="values")
    scores = _as_real_array(score, name="score")

    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if kind not in _PERCENTILE_METHODS:
        choices = ", ".join(sorted(_PERCENTILE_METHODS))
        raise ValueError(
            f"Invalid kind {kind!r}. Choose from: {choices}"
        )

    _validate_nan_policy(nan_policy)

    scalar_score = scores.ndim == 0
    original_score_shape = scores.shape
    scores = scores.reshape(-1)

    # Handle missing reference values according to the policy.
    reference_has_nan = np.any(np.isnan(arr))

    if reference_has_nan:
        if nan_policy == "raise":
            raise ValueError("values contains NaN values")

        if nan_policy == "propagate":
            result = np.full(scores.shape, np.nan, dtype=float)
            return float(result[0]) if scalar_score else result.reshape(
                original_score_shape
            )

        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        raise ValueError(
            "Cannot calculate percentile rank of an empty dataset"
        )

    # Sorting once is sufficient for every score.
    sorted_values = np.sort(arr, kind="stable")

    # NaN query scores produce NaN outputs.
    result = np.full(scores.shape, np.nan, dtype=float)
    valid_scores = ~np.isnan(scores)

    if np.any(valid_scores):
        query = scores[valid_scores]

        # left = number of observations strictly less than score.
        # right = number of observations less than or equal to score.
        left = np.searchsorted(sorted_values, query, side="left")
        right = np.searchsorted(sorted_values, query, side="right")

        n = sorted_values.size

        if kind == "weak":
            calculated = right / n * 100.0

        elif kind == "strict":
            calculated = left / n * 100.0

        elif kind == "mean":
            calculated = (left + right) / (2.0 * n) * 100.0

        else:  # kind == "rank"
            # If matches exist, average their one-based positions.
            # If absent, use the strict percentile rank.
            has_match = (right > left).astype(float)
            calculated = (
                left + right + has_match
            ) / (2.0 * n) * 100.0

        result[valid_scores] = calculated

    if scalar_score:
        return float(result[0])

    return result.reshape(original_score_shape)


# ============================================================
# 5. PUBLIC API
# ============================================================

__all__ = ["rankdata", "percentile_rank"]
