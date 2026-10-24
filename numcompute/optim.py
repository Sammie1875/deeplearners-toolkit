import numpy as np

def _validate_inputs(func, x, h, method):
    """
    Validate the common inputs used for finite-difference calculations.

    Parameters
    ----------
    func : callable
        Function to be differentiated.
    x : array_like
        Input point at which the derivative is estimated.
    h : float
        Positive, finite step size used for finite differences.
    method : str
        Finite-difference method: 'central' or 'forward'.

    Returns
    -------
    np.ndarray
        Input x converted to a floating-point NumPy array.

    Raises
    ------
    TypeError
        If func is not callable or h is not a real number.
    ValueError
        If method is invalid, h is not positive and finite,
        x is empty, or x contains non-finite values.
    """
    if not callable(func):
        raise TypeError("function must be callable")

    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    if (
        isinstance(h, (bool, np.bool_))
        or not isinstance(h, (int, float, np.integer, np.floating))
    ):
        raise TypeError("h must be a real number")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")

    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("x must not be empty")

    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")

    return x


def gradient(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar-valued function using finite differences.

    The gradient is calculated by perturbing each element of x individually.
    Central differences use function evaluations on both sides of each
    coordinate, while forward differences use the current point and a
    positively perturbed point.

    Parameters
    ----------
    f : callable
        Scalar-valued function that accepts a NumPy array and returns
        one finite numeric value.
    x : array_like
        Input point at which to estimate the gradient. Can have any
        non-empty shape, provided all elements are finite.
    h : float, optional
        Positive, finite step size used for numerical differentiation.
        Default is 1e-5.
    method : {'central', 'forward'}, optional
        Finite-difference method. 'central' uses the central-difference
        formula; 'forward' uses the forward-difference formula.
        Default is 'central'.

    Returns
    -------
    np.ndarray
        Estimated gradient with the same shape as x.

    Raises
    ------
    TypeError
        If f is not callable or h is not a real number.
    ValueError
        If method is invalid, h is not positive and finite, x is empty
        or contains non-finite values, or f returns a non-scalar or
        non-finite value.
    FloatingPointError
        If the calculated gradient contains non-finite values.

    Notes
    -----
    The result is an approximation to the true gradient. Its accuracy
    depends on the step size, floating-point precision, and behavior
    of f near x.
    """
    x = _validate_inputs(f, x, h, method)
    g = np.zeros_like(x)

    if method == "forward":
        f_x = np.asarray(f(x), dtype=float)
        if f_x.size != 1 or not np.isfinite(f_x).all():
            raise ValueError("f must return one finite scalar")
        f_x = f_x.item()

    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h

        if method == "central":
            x_minus = x.copy()
            x_minus.flat[i] -= h

            f_plus = np.asarray(f(x_plus), dtype=float)
            f_minus = np.asarray(f(x_minus), dtype=float)

            if f_plus.size != 1 or f_minus.size != 1:
                raise ValueError("f must return one scalar")

            if not np.isfinite(f_plus).all() or not np.isfinite(f_minus).all():
                raise ValueError("f must return finite values")

            g.flat[i] = (f_plus.item() - f_minus.item()) / (2 * h)

        else:
            f_plus = np.asarray(f(x_plus), dtype=float)

            if f_plus.size != 1 or not np.isfinite(f_plus).all():
                raise ValueError("f must return one finite scalar")

            g.flat[i] = (f_plus.item() - f_x) / h

    if not np.all(np.isfinite(g)):
        raise FloatingPointError("gradient contains non-finite values")

    return g


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian matrix of a vector-valued function using
    finite differences.

    Each column contains the partial derivatives of the output with
    respect to one element of the input. The output of F is flattened
    when constructing the Jacobian matrix.

    Parameters
    ----------
    F : callable
        Vector-valued function that accepts a NumPy array and returns
        a non-empty numeric array containing finite values.
    x : array_like
        Input point at which to estimate the Jacobian. Can have any
        non-empty shape, provided all elements are finite.
    h : float, optional
        Positive, finite step size used for numerical differentiation.
        Default is 1e-5.
    method : {'central', 'forward'}, optional
        Finite-difference method. 'central' uses the central-difference
        formula; 'forward' uses the forward-difference formula.
        Default is 'central'.

    Returns
    -------
    np.ndarray
        Jacobian matrix with shape (F(x).size, x.size).

    Raises
    ------
    TypeError
        If F is not callable or h is not a real number.
    ValueError
        If method is invalid, h is not positive and finite, x is empty
        or contains non-finite values, F returns empty or non-finite
        output, or the output shape changes between evaluations.
    FloatingPointError
        If the calculated Jacobian contains non-finite values.

    Notes
    -----
    The Jacobian is a numerical approximation. Its accuracy depends on
    the step size, floating-point precision, and behavior of F near x.
    """
    x = _validate_inputs(F, x, h, method)

    F_x = np.asarray(F(x), dtype=float)

    if F_x.size == 0 or not np.all(np.isfinite(F_x)):
        raise ValueError("F(x) must contain finite, non-empty values")

    F_x_flat = F_x.ravel()
    J = np.zeros((F_x_flat.size, x.size), dtype=float)

    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h

        F_plus = np.asarray(F(x_plus), dtype=float)

        if F_plus.shape != F_x.shape:
            raise ValueError("F must return a consistent output shape")

        if not np.all(np.isfinite(F_plus)):
            raise ValueError("F must return finite values")

        if method == "central":
            x_minus = x.copy()
            x_minus.flat[i] -= h
            F_minus = np.asarray(F(x_minus), dtype=float)

            if F_minus.shape != F_x.shape:
                raise ValueError("F must return a consistent output shape")

            if not np.all(np.isfinite(F_minus)):
                raise ValueError("F must return finite values")

            J[:, i] = (F_plus.ravel() - F_minus.ravel()) / (2 * h)

        else:
            J[:, i] = (F_plus.ravel() - F_x_flat) / h

    if not np.all(np.isfinite(J)):
        raise FloatingPointError("Jacobian contains non-finite values")

    return J

def gradient_batched(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient using batched finite-difference evaluations.

    Constructs perturbations for all input coordinates and passes the
    resulting batch of perturbed inputs to f. This requires f to support
    batched inputs and return one scalar output per batch element.

    Parameters
    ----------
    f : callable
        Function that accepts a batch of inputs with shape
        (n,) + x.shape and returns n scalar outputs, where n = x.size.
        For forward differences, f must also accept a batch of size one.
    x : array_like
        Input point at which to estimate the gradient. Its shape is
        preserved in the returned gradient.
    h : float, optional
        Positive finite-difference step size. Default is 1e-5.
    method : {'central', 'forward'}, optional
        'central' evaluates positive and negative perturbations.
        'forward' evaluates positive perturbations and the original point.
        Default is 'central'.

    Returns
    -------
    np.ndarray
        Estimated gradient with the same shape as x.

    Raises
    ------
    ValueError
        If method is invalid or the function outputs cannot be reshaped
        to the expected batch dimensions.

    Notes
    -----
    This implementation does not use _validate_inputs. Input validation,
    output finiteness, and step-size validation are therefore not
    guaranteed by this function.
    """   
    
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x_flat = x.ravel()
    n = x_flat.size

    perturbations = h * np.eye(n)
    X_plus = (x_flat + perturbations).reshape((n,) + shape)

    if method == "central":
        X_minus = (x_flat - perturbations).reshape((n,) + shape)
        g = (np.asarray(f(X_plus)).reshape(n)
             - np.asarray(f(X_minus)).reshape(n)) / (2 * h)

    elif method == "forward":
        f_plus = np.asarray(f(X_plus)).reshape(n)
        f_x = np.asarray(f(x.reshape((1,) + shape))).reshape(1)[0]
        g = (f_plus - f_x) / h

    else:
        raise ValueError("method must be 'central' or 'forward'")

    return g.reshape(shape)


def jacobian_batched(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian using batched finite-difference evaluations.

    Evaluates F on batches of positively and, for central differences,
    negatively perturbed inputs. The outputs are reshaped into a matrix
    whose columns correspond to input coordinates and whose rows
    correspond to flattened output coordinates.

    Parameters
    ----------
    F : callable
        Function that accepts a batch of inputs with shape
        (n,) + x.shape and returns outputs whose leading dimension is
        n, where n = x.size. For forward differences, F must also accept
        a batch of size one representing the original input.
    x : array_like
        Input point at which to estimate the Jacobian.
    h : float, optional
        Positive finite-difference step size. Default is 1e-5.
    method : {'central', 'forward'}, optional
        'central' evaluates positive and negative perturbations.
        'forward' evaluates positive perturbations and the original point.
        Default is 'central'.

    Returns
    -------
    np.ndarray
        Jacobian matrix with shape (m, n), where n = x.size and m is
        the number of output elements produced for one input.

    Raises
    ------
    ValueError
        If method is invalid or the function outputs cannot be reshaped
        to the expected batch dimensions.

    """
    
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x_flat = x.ravel()
    n = x_flat.size

    perturbations = h * np.eye(n)
    X_plus = (x_flat + perturbations).reshape((n,) + shape)

    if method == "central":
        X_minus = (x_flat - perturbations).reshape((n,) + shape)
        F_plus = np.asarray(F(X_plus)).reshape(n, -1)
        F_minus = np.asarray(F(X_minus)).reshape(n, -1)
        J = ((F_plus - F_minus) / (2 * h)).T

    elif method == "forward":
        F_plus = np.asarray(F(X_plus)).reshape(n, -1)
        F_x = np.asarray(F(x.reshape((1,) + shape))).reshape(1, -1)
        J = ((F_plus - F_x) / h).T

    else:
        raise ValueError("method must be 'central' or 'forward'")

    return J