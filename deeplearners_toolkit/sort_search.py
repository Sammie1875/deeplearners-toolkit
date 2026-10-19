import numpy as np

def stable_sort(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Return a stable sorted copy of an array. """
    
    return np.sort(values, axis=axis, kind="stable")


def multi_key_sort(
    data: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """Sort a two-dimensional array by multiple columns."""
    return np.lexsort(data, axis)

def binary_search(
    sorted_array: np.ndarray,
    x: object,
) -> tuple[int, bool]:
    """Find the insertion position for a value in a sorted array.
    """
    
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
    
    """Returns the top-k elements or their indices"""
    
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

    k = int(k)
    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}")

    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")

    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError("return_indices must be a boolean")
    
    start_idx = 0
    end_idx = k

    if largest:
        start_idx = values.size - k
        end_idx = values.size
        k = -k
    if return_indices:
        return np.argpartition(values, k)[start_idx:end_idx]
    
    return np.partition(values,k)[start_idx:end_idx]


def quick_select(values:np.ndarray, 
          k: int,
          largest: bool = True,
          return_index: bool = True):
    
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")

    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")

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
