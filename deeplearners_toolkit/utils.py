
"""
utils.py
========

Shared utility functions for the DeepLearners Toolkit.

Purpose
-------
Provide reusable helpers for numerical array validation,
shape checking, finite-value validation, safe division and
array metadata.

Dependencies
------------
Python standard library and NumPy only.

Design principles
-----------------
1. Keep functions small and reusable.
2. Raise clear exceptions for invalid inputs.
3. Do not silently change numerical data.
4. Return predictable NumPy arrays or Python values.
5. Avoid duplicating domain-specific logic from other modules.
"""

import numpy as np


# ============================================================
# 1. NUMERIC ARRAY CONVERSION
# ============================================================

def as_numeric_array(
    values,
    name="values",
    dtype=None,
    allow_complex=False,
):
    """
    Convert supported input into a NumPy numeric array.

    Parameters
    ----------
    values : array-like
        Input values such as a list, tuple or NumPy array.

    name : str, default="values"
        Name used in error messages.

    dtype : NumPy dtype or None, default=None
        Optional output dtype. If None, preserve the inferred
        numeric dtype where possible.

    allow_complex : bool, default=False
        Whether complex-valued numeric arrays are accepted.

    Returns
    -------
    np.ndarray
        A validated numeric array.

    Raises
    ------
    TypeError
        If the input is nonnumeric or has an unsupported dtype.

    Examples
    --------
    >>> as_numeric_array([1, 2, 3])
    array([1, 2, 3])

    >>> as_numeric_array([1.5, 2.5])
    array([1.5, 2.5])
    """
    try:
        arr = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be convertible to a numeric array"
        ) from exc

    # Reject strings, objects, Boolean arrays and other
    # nonnumeric types. Boolean data are not treated as numbers
    # by this toolkit's general numerical API.
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")

    if not allow_complex and np.issubdtype(
        arr.dtype, np.complexfloating
    ):
        raise TypeError(
            f"{name} must contain real values, not complex values"
        )

    return arr


# ============================================================
# 2. GENERAL ARRAY VALIDATION
# ============================================================

def validate_array(
    values,
    name="values",
    ndim=None,
    allow_empty=True,
    allow_nan=True,
    allow_inf=True,
    allow_complex=False,
):
    """
    Validate an array's numeric type, shape and value constraints.

    Parameters
    ----------
    values : array-like
        Input data.

    name : str
        Name used in error messages.

    ndim : int or None
        Required number of dimensions. None means any dimension.

    allow_empty : bool
        Whether arrays with zero elements are permitted.

    allow_nan : bool
        Whether NaN values are permitted.

    allow_inf : bool
        Whether positive and negative infinity are permitted.

    allow_complex : bool
        Whether complex-valued numeric data are permitted.

    Returns
    -------
    np.ndarray
        The validated array, without unnecessary dtype conversion.

    Raises
    ------
    TypeError
        If the array type is invalid.
    ValueError
        If dimensionality or value constraints are violated.
    """
    if ndim is not None:
        if (
            isinstance(ndim, (bool, np.bool_))
            or not isinstance(ndim, (int, np.integer))
            or ndim < 0
        ):
            raise ValueError("ndim must be a nonnegative integer")

    arr = as_numeric_array(
        values,
        name=name,
        allow_complex=allow_complex,
    )

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions; "
            f"received {arr.ndim}"
        )

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty")

    # NaN and infinity checks apply to floating-point arrays.
    # Complex arrays are also checked when explicitly allowed.
    if np.issubdtype(arr.dtype, np.inexact):
        if not allow_nan and np.any(np.isnan(arr)):
            raise ValueError(f"{name} must not contain NaN values")

        if not allow_inf and np.any(np.isinf(arr)):
            raise ValueError(
                f"{name} must not contain infinite values"
            )

    return arr


# ============================================================
# 3. FINITE-VALUE VALIDATION
# ============================================================

def check_finite(values, name="values"):
    """
    Require all array values to be finite.

    Finite means that no element is NaN, positive infinity or
    negative infinity.

    Returns
    -------
    np.ndarray
        The validated array.
    """
    arr = as_numeric_array(
        values,
        name=name,
        allow_complex=False,
    )

    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

    return arr


# ============================================================
# 4. SHAPE VALIDATION
# ============================================================

def validate_matching_shapes(*arrays, names=None):
    """
    Validate that two or more arrays have exactly matching shapes.

    Useful when comparing actual and predicted values, checking
    corresponding numerical arrays, or validating elementwise
    calculations.

    Parameters
    ----------
    *arrays
        Two or more array-like objects.

    names : sequence of str or None
        Optional names for clearer error messages.

    Returns
    -------
    tuple of np.ndarray
        Validated arrays in their original order.

    Raises
    ------
    ValueError
        If fewer than two arrays are provided or shapes differ.
    TypeError
        If any input is not numeric.

    Examples
    --------
    >>> a, b = validate_matching_shapes([1, 2], [3, 4])
    >>> a.shape == b.shape
    True
    """
    if len(arrays) < 2:
        raise ValueError(
            "At least two arrays are required for shape comparison"
        )

    if names is None:
        names = [f"array_{i + 1}" for i in range(len(arrays))]
    elif len(names) != len(arrays):
        raise ValueError(
            "names must contain one name per input array"
        )

    converted = tuple(
        as_numeric_array(arr, name=name)
        for arr, name in zip(arrays, names)
    )

    expected_shape = converted[0].shape

    for name, arr in zip(names[1:], converted[1:]):
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arr.shape}; "
                f"expected {expected_shape}"
            )

    return converted


# ============================================================
# 5. SAFE DIVISION
# ============================================================

def safe_divide(numerator, denominator, zero_policy="raise"):
    """
    Divide numerical values with explicit zero-denominator handling.

    Parameters
    ----------
    numerator : array-like
        Numerator values.

    denominator : array-like
        Denominator values. NumPy broadcasting is supported.

    zero_policy : str, default="raise"
        "raise":
            Raise ZeroDivisionError if any denominator is zero.

        "nan":
            Return NaN wherever the denominator is zero.

        "zero":
            Return zero wherever the denominator is zero.

    Returns
    -------
    np.ndarray
        The division result. The output shape follows NumPy's
        broadcasting rules.

    Notes
    -----
    Nonzero divisions use normal NumPy arithmetic. This function
    does not silently replace NaNs or infinities produced by
    other arithmetic operations.
    """
    if zero_policy not in {"raise", "nan", "zero"}:
        raise ValueError(
            "zero_policy must be 'raise', 'nan' or 'zero'"
        )

    num = as_numeric_array(numerator, name="numerator")
    den = as_numeric_array(denominator, name="denominator")

    try:
        num, den = np.broadcast_arrays(num, den)
    except ValueError as exc:
        raise ValueError(
            "numerator and denominator shapes are not broadcastable"
        ) from exc

    zero_mask = den == 0

    if zero_policy == "raise" and np.any(zero_mask):
        raise ZeroDivisionError(
            "Division by zero encountered in denominator"
        )

    # Use floating-point output so NaN can be represented.
    # Complex inputs are rejected by the general array validator.
    try:
        output = np.full(num.shape, np.nan, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Inputs must support real numerical division"
        ) from exc

    nonzero_mask = ~zero_mask

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            num,
            den,
            out=output,
            where=nonzero_mask,
            casting="unsafe",
        )

    if zero_policy == "zero":
        output[zero_mask] = 0.0

    return output


# ============================================================
# 6. ARRAY INFORMATION
# ============================================================

def get_array_info(values, name="values"):
    """
    Return basic metadata about a numeric array.

    This is useful for debugging, logging and inspecting the
    numerical data used by the toolkit.

    Returns
    -------
    dict
        Contains name, shape, ndim, size, dtype and nbytes.
    """
    arr = as_numeric_array(values, name=name, allow_complex=True)

    return {
        "name": name,
        "shape": arr.shape,
        "ndim": arr.ndim,
        "size": arr.size,
        "dtype": str(arr.dtype),
        "nbytes": arr.nbytes,
    }


# ============================================================
# 7. PUBLIC API
# ============================================================

__all__ = [
    "as_numeric_array",
    "validate_array",
    "check_finite",
    "validate_matching_shapes",
    "safe_divide",
    "get_array_info",
]
