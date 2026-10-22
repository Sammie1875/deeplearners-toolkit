import numpy as np


def gradient(f, x, h=1e-5, method="central"):
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


x = np.array([2.0, 3.0])

# Scalar-valued function
f = lambda x: x[0]**2 + 3 * x[1]**2
print(gradient(f, x))
print(gradient(f, x, method="forward"))
# Approximately [4., 18.]

# Vector-valued function
F = lambda x: np.array([
    x[0]**2 + x[1],
    x[0] * x[1]
])
print(jacobian(F, x))
print(jacobian(F, x, method="forward"))
# Approximately:
# [[4., 1.],
#  [3., 2.]]


def gradient_batched(f, x, h=1e-5, method="central"):
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

def f(X):
    return np.sum(X**2, axis=-1)

def F(X):
    return np.column_stack((
        X[:, 0]**2 + X[:, 1],
        X[:, 0] * X[:, 1]
    ))

x = np.array([2.0, 3.0])

print(gradient_batched(f, x))
# [4. 6.]

print(gradient_batched(f, x, method="forward"))
# [4.00001 6.00001]

print(jacobian_batched(F, x))
# [[4. 1.]
#  [3. 2.]]

print(jacobian_batched(F, x, method="forward"))
# [[4.00001 1.     ]
#  [3.      2.     ]]