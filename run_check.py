import sys
import os

# Add the project root directory to Python's import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import numpy as np

from deeplearners_toolkit.io import read_csv
from deeplearners_toolkit.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

print("Imports successful!")

# Test StandardScaler
X = np.array([[1, 2], [3, 4]])
scaler = StandardScaler()
print("StandardScaler:", scaler.fit_transform(X))

# Test MinMaxScaler
mm = MinMaxScaler()
print("MinMaxScaler:", mm.fit_transform(X))

# Test OneHotEncoder
X_cat = np.array([[1, 2], [1, 3]])
encoder = OneHotEncoder()
print("OneHotEncoder:", encoder.fit_transform(X_cat))
