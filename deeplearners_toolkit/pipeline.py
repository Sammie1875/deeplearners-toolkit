class Pipeline:
    """Chain named preprocessors and an optional final estimator.

    Each preprocessor must implement fit(X) and transform(X).
    A final estimator must implement fit(X, y) and predict(X).
    """

    def __init__(self, steps):
        """Store an ordered sequence of (name, component) pairs."""
        self.steps = list(steps)
        self._fitted = False
        self._final_is_estimator = False

    def fit_transform(self, X):
        """Fit each preprocessing step and return the transformed data."""
        X_current = X

        for name, step in self.steps:
            step.fit(X_current)
            X_current = step.transform(X_current)

        self._fitted = True
        self._final_is_estimator = False
        return X_current

    def fit(self, X, y):
        """Fit preprocessors in order, then fit the final estimator if present.

        If the final component has predict(), it is treated as an estimator
        and receives transformed X and y. Otherwise, it is treated as a
        preprocessor.
        """
        if not self.steps:
            raise ValueError("steps must contain at least one component")

        X_current = X

        for name, step in self.steps[:-1]:
            step.fit(X_current)
            X_current = step.transform(X_current)

        final_step = self.steps[-1][1]

        if hasattr(final_step, "predict"):
            final_step.fit(X_current, y)
            self._final_is_estimator = True
        else:
            final_step.fit(X_current)
            X_current = final_step.transform(X_current)
            self._final_is_estimator = False

        self._fitted = True
        return self

    def transform(self, X):
        """Transform X using already-fitted preprocessing steps."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        X_current = X
        steps_to_apply = self.steps[:-1] if self._final_is_estimator else self.steps

        for name, step in steps_to_apply:
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        """Transform X and return predictions from the final estimator."""
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before predict")

        if not self._final_is_estimator:
            raise AttributeError(
                "The final pipeline step is a preprocessor, not an estimator"
            )

        X_current = X

        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)

        final_step = self.steps[-1][1]
        return final_step.predict(X_current)
