import numpy as np

def read_csv(path, delimiter=",", missing_value=np.nan, dtype=float):
    """
    Read a CSV file into a NumPy array.

    Parameters
    ----------
    path : str
        File path to the CSV.
    delimiter : str
        Column delimiter.
    missing_value : float
        Value to replace missing entries.
    dtype : type
        Data type of the output array.

    Returns
    -------
    np.ndarray
        Loaded data.
    """
    return np.genfromtxt(
        path,
        delimiter=delimiter,
        filling_values=missing_value,
        dtype=dtype
    )
