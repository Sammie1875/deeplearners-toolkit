import numpy as np

class StandardScaler:
    """
    Standardise features by removing the mean and scaling to unit variance.
    """

    def fit(self, X):
        # Compute mean and standard deviation for each column
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        # Apply standardisation: (X - mean) / std
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        # Convenience method: fit then transform
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale features to a given range (default: 0 to 1).
    """

    def __init__(self, feature_range=(0, 1)):
        # Desired output range
        self.min_val, self.max_val = feature_range

    def fit(self, X):
        # Compute min and max for each column
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        # Scale to [0, 1], then to desired range
        scaled = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        return scaled * (self.max_val - self.min_val) + self.min_val

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Convert categorical columns into one-hot encoded vectors.
    """

    def fit(self, X):
        # Identify unique categories for each column
        self.categories_ = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        # Convert each column into one-hot encoded vectors
        outputs = []
        for col, cats in zip(X.T, self.categories_):
            onehot = (col[:, None] == cats[None, :]).astype(int)
            outputs.append(onehot)
        return np.hstack(outputs)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
