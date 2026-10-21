import numpy as np


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar-valued function.

    Parameters
    ----------
    f : callable
        Function that accepts a NumPy array and returns a scalar.
    x : np.ndarray
        Input point.
    h : float
        Finite-difference step size.
    method : str
        'central' or 'forward'.

    Returns
    -------
    np.ndarray
        Gradient with the same shape as x.
    """
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")
    
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)

    f_x = f(x) if method == "forward" else None

    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h

        if method == "central":
            x_minus = x.copy()
            x_minus.flat[i] -= h
            g.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)

        else:
            g.flat[i] = (f(x_plus) - f_x) / h

    return g


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian of a vector-valued function.

    Parameters
    ----------
    F : callable
        Function that accepts a NumPy array and returns an array.
    x : np.ndarray
        Input point.
    h : float
        Finite-difference step size.
    method : str
        'central' or 'forward'.

    Returns
    -------
    np.ndarray
        Jacobian with shape (F(x).size, x.size).
    """
    x = np.asarray(x, dtype=float)
    F_x = np.asarray(F(x), dtype=float)
    J = np.zeros((F_x.size, x.size), dtype=float)

    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h

        if method == "central":
            x_minus = x.copy()
            x_minus.flat[i] -= h

            F_plus = np.asarray(F(x_plus), dtype=float).ravel()
            F_minus = np.asarray(F(x_minus), dtype=float).ravel()

            J[:, i] = (F_plus - F_minus) / (2 * h)

        elif method == "forward":
            F_plus = np.asarray(F(x_plus), dtype=float).ravel()
            J[:, i] = (F_plus - F_x.ravel()) / h

        else:
            raise ValueError("method must be 'central' or 'forward'")

    return J