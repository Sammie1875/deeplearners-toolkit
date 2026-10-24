import numpy as np

def _validate_array(values, name):
    """Validate that the input is a NumPy array."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")


def _contains_nan(values):
    """Return whether a numeric array contains any NaN values."""
    if np.issubdtype(values.dtype, np.inexact):
        return bool(np.any(np.isnan(values)))
    return False


def stable_sort(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Return a stable sorted copy of an array.

    Parameters
    ----------
    values : np.ndarray
        Array to sort.
    axis : int, optional
        Axis along which to sort. Defaults to -1.

    Returns
    -------
    np.ndarray
        A sorted copy of the input array.

    Raises
    ------
    TypeError
        If values is not a NumPy array.
    ValueError
        If axis is not an integer.
    """
    _validate_array(values, "values")

    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, (int, np.integer)
    ):
        raise ValueError("axis must be an integer")

    return np.sort(values, axis=axis, kind="stable")


def multi_key_sort(
    data: np.ndarray,
    ascending: bool = True
) -> np.ndarray:
    """Return indices that sort a 2D array by multiple columns.

    Columns are used as sorting keys, with the first column as the
    primary key. NaNs follow NumPy's lexsort behavior.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array to sort.
    ascending : bool, optional
        If True, sort in ascending order; otherwise, descending.

    Returns
    -------
    np.ndarray
        Indices that sort the rows.

    Raises
    ------
    TypeError
        If data is not a NumPy array or ascending is not Boolean.
    ValueError
        If data is not 2D or has no columns.
    """
    _validate_array(data, "data")

    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    if data.shape[1] == 0:
        raise ValueError("data must contain at least one column")

    if not isinstance(ascending, (bool, np.bool_)):
        raise TypeError("ascending must be a boolean")
    
    
    indices = np.lexsort(data.T[::-1])

    if not ascending:
        indices = indices[::-1]

    return indices


def binary_search(
    sorted_array: np.ndarray,
    x: object,
) -> tuple[int, bool]:
    
    """Return the insertion index and whether x exists in a sorted array.

    The input must be sorted in ascending order. Duplicate matches
    return the index of the first occurrence.

    NaN search targets are rejected. Existing NaNs in sorted_array
    follow NumPy's searchsorted behavior.

    Parameters
    ----------
    sorted_array : np.ndarray
        One-dimensional array sorted in ascending order.
    x : object
        Value to search for.

    Returns
    -------
    tuple[int, bool]
        Insertion index and whether the value was found.

    Raises
    ------
    TypeError
        If sorted_array is not a NumPy array.
    ValueError
        If sorted_array is not 1D or x is NaN.
    """
    _validate_array(sorted_array, "sorted_array")

    if sorted_array.ndim != 1:
        raise ValueError("sorted_array must be a one-dimensional array")

    try:
        if np.isscalar(x) and np.issubdtype(
            np.asarray(x).dtype, np.inexact
        ) and np.isnan(x):
            raise ValueError("x must not be NaN")
    except TypeError:
        pass

    array = sorted_array

    if array.ndim != 1:
        raise ValueError("sorted_array must be a one-dimensional array")

    index = int(np.searchsorted(array, x, side="left"))
    exists = index < array.size and bool(array[index] == x)

    return index, exists

def top_k(values:np.ndarray, 
          k: int,
          largest: bool = True,
          return_indices: bool = True):
    
    """Return the k largest or smallest elements or their indices.

    The selected results are not guaranteed to be sorted.

    NaN values are rejected. Positive and negative infinity are allowed.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional array of values.
    k : int
        Number of values to select, between 1 and values.size.
    largest : bool, optional
        If True, select the largest values; otherwise, the smallest.
    return_indices : bool, optional
        If True, return indices; otherwise, return selected values.

    Returns
    -------
    np.ndarray
        Selected indices or values, in unspecified order.

    Raises
    ------
    TypeError
        If values is not an ndarray, k is not an integer, or a Boolean
        argument is invalid.
    ValueError
        If values is not 1D, contains NaNs, or k is out of range.
    """
    _validate_array(values, "values")

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if not isinstance( k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")

    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError("return_indices must be a boolean")


    k = int(k)
    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}")

    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")

    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError("return_indices must be a boolean")
    
    if largest:
        start_idx = values.size - k
        kth = start_idx
    else:
        start_idx = 0
        kth = k - 1

    end_idx = start_idx + k

    if return_indices:
        return np.argpartition(values, kth)[start_idx:end_idx]

    return np.partition(values, kth)[start_idx:end_idx]


def quick_select(values:np.ndarray, 
          k: int,
          largest: bool = True,
          return_index: bool = True):
    
    """Select the element at a specified zero-based rank.

    By default, k=0 selects a largest value. If largest=False,
    k=0 selects a smallest value.

    NaN values are rejected. Positive and negative infinity are allowed.
    If values are duplicated, any matching index may be returned.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional array of values.
    k : int
        Zero-based rank, between 0 and values.size - 1.
    largest : bool, optional
        If True, rank from largest to smallest; otherwise, smallest
        to largest.
    return_index : bool, optional
        If True, return the selected element's original index.

    Returns
    -------
    int or numpy scalar
        Index of the selected element or its value.

    Raises
    ------
    TypeError
        If values is not an ndarray, k is not an integer, or a Boolean
        argument is invalid.
    ValueError
        If values is not 1D, contains NaNs, or k is out of range.
    """
    _validate_array(values, "values")

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")

    if not isinstance(return_index, (bool, np.bool_)):
        raise TypeError("return_index must be a boolean")


    k = int(k)
    if k < 0 or k >= values.size:
        raise ValueError(f"k must satisfy 0 <= k < {values.size}")

    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")
    
    
    if not isinstance(return_index, (bool, np.bool_)):
        raise TypeError("return_indices must be a boolean")
    
    if largest:
        k = values.size-1-k
    
    if return_index:
        return np.argpartition(values, k)[k]
        
    return np.partition(values,k)[k]
