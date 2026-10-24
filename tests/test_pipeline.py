
import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)


class DummyEstimator:
    """Simple estimator for testing Pipeline."""

    def fit(self, X, y):
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.sum(X, axis=1)


class DummyPreprocessor:
    """Simple preprocessor that adds 1 to every element."""

    def fit(self, X):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X + 1


# --------------------------------------------------
# fit_transform tests
# --------------------------------------------------

def test_fit_transform_single_preprocessor():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipeline = Pipeline([("scale", StandardScaler())])
    result = pipeline.fit_transform(X)

    expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(result, expected)
    assert pipeline._fitted


def test_fit_transform_multiple_preprocessors():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipeline = Pipeline([
        ("add", DummyPreprocessor()),
        ("scale", MinMaxScaler()),
    ])

    result = pipeline.fit_transform(X)

    expected = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(result, expected)
    assert pipeline._fitted
    assert not pipeline._final_is_estimator


def test_fit_transform_one_hot_encoder():
    X = np.array([
        ["red", "S"],
        ["blue", "M"],
        ["red", "M"],
    ])

    pipeline = Pipeline([("encode", OneHotEncoder())])
    result = pipeline.fit_transform(X)

    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ])

    assert np.array_equal(result, expected)


# --------------------------------------------------
# fit tests
# --------------------------------------------------

def test_fit_with_estimator():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 1])

    estimator = DummyEstimator()
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", estimator),
    ])

    result = pipeline.fit(X, y)

    expected_X = StandardScaler().fit_transform(X)

    assert result is pipeline
    assert pipeline._fitted
    assert pipeline._final_is_estimator
    assert np.allclose(estimator.X_, expected_X)
    assert np.array_equal(estimator.y_, y)


def test_fit_with_final_preprocessor():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("minmax", MinMaxScaler()),
    ])

    result = pipeline.fit(X, np.array([0, 1]))

    expected = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert result is pipeline
    assert pipeline._fitted
    assert not pipeline._final_is_estimator
    assert np.allclose(pipeline.transform(X), expected)


def test_fit_with_empty_steps_raises_error():
    pipeline = Pipeline([])

    try:
        pipeline.fit(np.array([[1.0]]), np.array([0]))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------
# transform tests
# --------------------------------------------------

def test_transform_before_fit_raises_error():
    pipeline = Pipeline([("scale", StandardScaler())])

    try:
        pipeline.transform(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_transform_after_fit_transform():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_test = np.array([[5.0, 6.0]])

    pipeline = Pipeline([("scale", StandardScaler())])
    pipeline.fit_transform(X_train)

    result = pipeline.transform(X_test)
    expected = np.array([[3.0, 3.0]])

    assert np.allclose(result, expected)


def test_transform_does_not_transform_final_estimator():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 1])

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyEstimator()),
    ])

    pipeline.fit(X, y)
    result = pipeline.transform(X)

    expected = StandardScaler().fit_transform(X)
    assert np.allclose(result, expected)


# --------------------------------------------------
# predict tests
# --------------------------------------------------

def test_predict_with_estimator():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 1])

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyEstimator()),
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    expected_X = StandardScaler().fit_transform(X)
    expected = np.sum(expected_X, axis=1)

    assert np.allclose(predictions, expected)


def test_predict_before_fit_raises_error():
    pipeline = Pipeline([("model", DummyEstimator())])

    try:
        pipeline.predict(np.array([[1.0, 2.0]]))
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_predict_without_estimator_raises_error():
    pipeline = Pipeline([("scale", StandardScaler())])
    pipeline.fit(np.array([[1.0], [3.0]]), np.array([0, 1]))

    try:
        pipeline.predict(np.array([[2.0]]))
        assert False, "Expected AttributeError"
    except AttributeError:
        pass


# --------------------------------------------------
# Run all tests
# --------------------------------------------------

def run_all_tests():
    test_fit_transform_single_preprocessor()
    test_fit_transform_multiple_preprocessors()
    test_fit_transform_one_hot_encoder()

    test_fit_with_estimator()
    test_fit_with_final_preprocessor()
    test_fit_with_empty_steps_raises_error()

    test_transform_before_fit_raises_error()
    test_transform_after_fit_transform()
    test_transform_does_not_transform_final_estimator()

    test_predict_with_estimator()
    test_predict_before_fit_raises_error()
    test_predict_without_estimator_raises_error()

    print("All pipeline tests passed!")


if __name__ == "__main__":
    run_all_tests()
