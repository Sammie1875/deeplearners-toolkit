import numpy as np
from deeplearners_toolkit.preprocessing import StandardScaler

def test_standard_scaler():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape
