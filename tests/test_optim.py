
import numpy as np
from numcompute.optim import gradient, jacobian


# -------------------------
# Gradient tests
# -------------------------

def test_gradient_quadratic_central():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([2.0, 3.0])

    result = gradient(f, x, method="central")

    expected = np.array([4.0, 6.0])

    assert np.allclose(result, expected, atol=1e-5)


def test_gradient_quadratic_forward():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([2.0, 3.0])

    result = gradient(f, x, method="forward")

    expected = np.array([4.0, 6.0])

    assert np.allclose(result, expected, atol=1e-4)


def test_gradient_linear():
    def f(x):
        return 3 * x[0] + 5 * x[1]

    x = np.array([1.0, 2.0])

    result = gradient(f, x)

    expected = np.array([3.0, 5.0])

    assert np.allclose(result, expected, atol=1e-5)


def test_gradient_constant_function():
    def f(x):
        return 7.0

    x = np.array([1.0, 2.0, 3.0])

    result = gradient(f, x)

    expected = np.zeros_like(x)

    assert np.allclose(result, expected, atol=1e-5)


def test_gradient_multidimensional_input():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = gradient(f, x)

    expected = np.array([[2.0, 4.0], [6.0, 8.0]])

    assert result.shape == x.shape
    assert np.allclose(result, expected, atol=1e-5)


def test_gradient_custom_step_size():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([2.0, 3.0])

    result = gradient(f, x, h=1e-4)

    expected = np.array([4.0, 6.0])

    assert np.allclose(result, expected, atol=1e-5)


def test_gradient_invalid_method():
    def f(x):
        return np.sum(x ** 2)

    x = np.array([1.0, 2.0])

    try:
        gradient(f, x, method="invalid")
        assert False, "Expected ValueError for invalid method"
    except ValueError:
        pass


# -------------------------
# Jacobian tests
# -------------------------

def test_jacobian_linear_function_central():
    def F(x):
        return np.array([
            2 * x[0] + 3 * x[1],
            4 * x[0] - x[1]
        ])

    x = np.array([1.0, 2.0])

    result = jacobian(F, x, method="central")

    expected = np.array([
        [2.0, 3.0],
        [4.0, -1.0]
    ])

    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_linear_function_forward():
    def F(x):
        return np.array([
            2 * x[0] + 3 * x[1],
            4 * x[0] - x[1]
        ])

    x = np.array([1.0, 2.0])

    result = jacobian(F, x, method="forward")

    expected = np.array([
        [2.0, 3.0],
        [4.0, -1.0]
    ])

    assert np.allclose(result, expected, atol=1e-4)


def test_jacobian_quadratic_function():
    def F(x):
        return np.array([
            x[0] ** 2,
            x[0] * x[1],
            x[1] ** 2
        ])

    x = np.array([2.0, 3.0])

    result = jacobian(F, x)

    expected = np.array([
        [4.0, 0.0],
        [3.0, 2.0],
        [0.0, 6.0]
    ])

    assert result.shape == (3, 2)
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_scalar_output():
    def F(x):
        return np.array([x[0] ** 2 + x[1] ** 2])

    x = np.array([2.0, 3.0])

    result = jacobian(F, x)

    expected = np.array([[4.0, 6.0]])

    assert result.shape == (1, 2)
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_constant_function():
    def F(x):
        return np.array([5.0, 10.0, -2.0])

    x = np.array([1.0, 2.0])

    result = jacobian(F, x)

    expected = np.zeros((3, 2))

    assert result.shape == (3, 2)
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_multidimensional_input():
    def F(x):
        return np.array([
            np.sum(x),
            np.sum(x ** 2)
        ])

    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = jacobian(F, x)

    expected = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 4.0, 6.0, 8.0]
    ])

    assert result.shape == (2, 4)
    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_custom_step_size():
    def F(x):
        return np.array([x[0] ** 2, x[1] ** 2])

    x = np.array([2.0, 3.0])

    result = jacobian(F, x, h=1e-4)

    expected = np.array([
        [4.0, 0.0],
        [0.0, 6.0]
    ])

    assert np.allclose(result, expected, atol=1e-5)


def test_jacobian_invalid_method():
    def F(x):
        return np.array([x[0] ** 2, x[1] ** 2])

    x = np.array([1.0, 2.0])

    try:
        jacobian(F, x, method="invalid")
        assert False, "Expected ValueError for invalid method"
    except ValueError:
        pass


if __name__ == "__main__":
    test_gradient_quadratic_central()
    test_gradient_quadratic_forward()
    test_gradient_linear()
    test_gradient_constant_function()
    test_gradient_multidimensional_input()
    test_gradient_custom_step_size()
    test_gradient_invalid_method()

    test_jacobian_linear_function_central()
    test_jacobian_linear_function_forward()
    test_jacobian_quadratic_function()
    test_jacobian_scalar_output()
    test_jacobian_constant_function()
    test_jacobian_multidimensional_input()
    test_jacobian_custom_step_size()
    test_jacobian_invalid_method()

    print("All 15 optim tests passed!")

