import numpy as np

def read_csv(path, delimiter=",", missing_value=np.nan, dtype=float):
    """
    Load a CSV file into a NumPy array.
    """

    # Use NumPy's genfromtxt for simple CSV loading with:
    # - custom delimiters
    # - missing value handling
    # - type conversion
    return np.genfromtxt(
        path,
        delimiter=delimiter,
        filling_values=missing_value,
        dtype=dtype
    )
