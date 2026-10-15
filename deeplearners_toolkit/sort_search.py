import numpy as np

def stable_sort(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Return a stable sorted copy of an array. """
    
    return np.sort(values, axis=axis, kind="stable")


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