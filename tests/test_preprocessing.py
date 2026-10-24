import numpy as np
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


# ---------------------------------------------------------
# StandardScaler Tests (3 tests)
# ---------------------------------------------------------

def test_standard_scaler_shape_preserved():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape


def test_standard_scaler_mean_zero():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Mean of each column should be ~0
    assert np.allclose(np.mean(X_scaled, axis=0), 0)


def test_standard_scaler_std_one():
    X = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Std of each column should be ~1
    assert np.allclose(np.std(X_scaled, axis=0), 1)


# ---------------------------------------------------------
# MinMaxScaler Tests (3 tests)
# ---------------------------------------------------------

def test_minmax_scaler_range_zero_one():
    X = np.array([[1, 2], [3, 4]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.min() == 0
    assert X_scaled.max() == 1


def test_minmax_scaler_custom_range():
    X = np.array([[1, 2], [3, 4]])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.min() == -1
    assert X_scaled.max() == 1


def test_minmax_scaler_shape_preserved():
    X = np.array([[1, 2], [3, 4]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape


# ---------------------------------------------------------
# OneHotEncoder Tests (3 tests)
# ---------------------------------------------------------

def test_onehot_encoder_column_count():
    X = np.array([[1, 2], [1, 3]])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    # Column 1 has 1 unique value, column 2 has 2 unique values → total 3 columns
    assert X_encoded.shape == (2, 3)


def test_onehot_encoder_correct_encoding():
    X = np.array([[1, 2], [1, 3]])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # First column: only category "1" → always encoded as [1]
    # Second column: categories {2, 3} → encoded as [1,0] for 2 and [0,1] for 3
    expected_first_row = np.array([1, 1, 0])  # 1 in col1, 2 in col2
    assert np.array_equal(X_encoded[0], expected_first_row)


def test_onehot_encoder_repeated_categories():
    X = np.array([[5, 5], [5, 5]])
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # Only one unique category per column → 2 columns total
    assert X_encoded.shape == (2, 2)
    assert np.all(X_encoded == np.array([[1, 1], [1, 1]]))
