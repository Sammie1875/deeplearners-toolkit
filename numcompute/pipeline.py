class Pipeline:
    """Chain named preprocessors and an optional final estimator.

    A pipeline applies components sequentially, passing the output of each
    component to the next. Preprocessors must implement ``fit(X)`` and
    ``transform(X)``. A final estimator must implement ``fit(X, y)`` and
    ``predict(X)``.

    Attributes:
        steps (list[tuple[str, object]]): Ordered sequence of component names
            and their corresponding preprocessing or estimation objects.
        _fitted (bool): Whether the pipeline has been fitted.
        _final_is_estimator (bool): Whether the final component is treated
            as an estimator rather than a preprocessor.
    """

    def __init__(self, steps):
        """Store an ordered sequence of (name, component) pairs."""
        self.steps = list(steps)
        self._fitted = False
        self._final_is_estimator = False

    def fit_transform(self, X):
        """Fit the pipeline and return the transformed input.

    Fits and applies each preprocessing step in order. The final
    component must support transformation through either
    fit_transform() or fit() and transform().

    Args:
        X (numpy.ndarray): Input data to fit and transform.

    Returns:
        numpy.ndarray: Transformed input data.

    Raises:
        ValueError: If the pipeline contains no components.
        AttributeError: If a component does not support transformation.
    """
        if not self.steps:
            raise ValueError("steps must contain at least one component")

        X_current = X

        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X_current = step.fit_transform(X_current)
            elif hasattr(step, "fit") and hasattr(step, "transform"):
                step.fit(X_current)
                X_current = step.transform(X_current)
            else:
                raise AttributeError(
                    f"Step '{name}' does not support transformation"
                )

        self._fitted = True
        self._final_is_estimator = False
        return X_current

    def fit(self, X, y):
        """Fit the pipeline's preprocessing steps and optional estimator.

        Intermediate components are fitted and applied sequentially. If the
        final component has a ``predict`` method, it is treated as an
        estimator and fitted using the transformed input and target values.
        Otherwise, the final component is treated as a preprocessor and is
        fitted and applied to the input.

        Args:
            X (numpy.ndarray): Input data used to fit the pipeline.
            y (numpy.ndarray): Target values passed to the final estimator,
                if one is present.

        Returns:
            Pipeline: The fitted pipeline instance.

        Raises:
            ValueError: If the pipeline contains no components.
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
        """Transform input data using the fitted preprocessing components.

        Applies each relevant component's ``transform`` method in pipeline
        order, without fitting the components again.

        Args:
            X (numpy.ndarray): Input data to transform.

        Returns:
            numpy.ndarray: Transformed input data.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        X_current = X
        steps_to_apply = self.steps[:-1] if self._final_is_estimator else self.steps

        for name, step in steps_to_apply:
            X_current = step.transform(X_current)

        return X_current

    def predict(self, X):
        """Generate predictions using the fitted final estimator.

        Transforms the input through all preprocessing components preceding
        the final estimator, then calls the estimator's ``predict`` method.

        Args:
            X (numpy.ndarray): Input data for which predictions are required.

        Returns:
            numpy.ndarray: Predictions produced by the final estimator.

        Raises:
            RuntimeError: If the pipeline has not been fitted.
            AttributeError: If the final component is not an estimator.
            """
        
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
