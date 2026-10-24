import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Proportion of correct predictions.

    Raises
    ------
    ValueError
        If the shapes of y_true and y_pred do not match.
        or contain NaN values.
        
    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(1)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true and predicted labels must match.")
    
    if y_true.size == 0:
        raise ValueError("Input arrays must not be empty.")
    
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute the confusion matrix for classification.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : array-like or None, optional
        Class labels to include. If None, labels are
        determined automatically from y_true and y_pred.

    Returns
    -------
    np.ndarray
        Confusion matrix where rows represent true labels
        and columns represent predicted labels.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes
        or are empty.

    Notes
    -----
    Time complexity: O(n * c), where c is the number of classes.
    Space complexity: O(c^2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    # Vectorised counting for each true/predicted label combination
    true_matches = y_true[:, None] == labels
    pred_matches = y_pred[:, None] == labels

    matrix = true_matches.astype(int).T @ pred_matches.astype(int)

    return matrix

def precision(y_true, y_pred, positive_label=1):
    """
    Calculate precision for binary classification.

    Precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        Precision score. Returns 0.0 if there are
        no predicted positive samples.

    Raises
    ------
    ValueError
        If input shapes differ or inputs are empty.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    true_positive = np.sum(
        (y_true == positive_label) & (y_pred == positive_label)
    )

    false_positive = np.sum(
        (y_true != positive_label) & (y_pred == positive_label)
    )

    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def recall(y_true, y_pred, positive_label=1):
    """
    Calculate recall for binary classification.

    Recall = TP / (TP + FN)

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        Recall score. Returns 0.0 if there are
        no actual positive samples.

    Raises
    ------
    ValueError
        If input shapes differ or inputs are empty.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    true_positive = np.sum(
        (y_true == positive_label) & (y_pred == positive_label)
    )

    false_negative = np.sum(
        (y_true == positive_label) & (y_pred != positive_label)
    )

    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def f1(y_true, y_pred, positive_label=1):
    """
    Calculate the F1 score for binary classification.

    F1 is the harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : int or str, default=1
        Label considered as the positive class.

    Returns
    -------
    float
        F1 score.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)

def mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        True numerical values.
    y_pred : array-like
        Predicted numerical values.

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If y_true and y_pred have different shapes
        or contain no values.

    Notes
    -----
    Time complexity: O(n)
    Space complexity: O(n)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Input arrays cannot be empty")

    return np.mean((y_true - y_pred) ** 2)