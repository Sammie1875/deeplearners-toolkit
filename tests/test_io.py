import numpy as np
from deeplearners_toolkit.io import read_csv

def test_read_csv_basic():
    # Load a sample CSV file using our custom reader
    data = read_csv("tests/sample.csv", delimiter=",")

    # The output should always be a NumPy array
    assert isinstance(data, np.ndarray)

    # Basic shape check (assuming sample.csv has 2 rows and 2 columns)
    assert data.shape == (2, 2)

    # Check a couple of values to ensure correct loading
    assert data[0, 0] == 1
    assert data[1, 1] == 4
