
import numpy as np
import pytest

from utils import (
    as_numeric_array,
    validate_array,
    check_finite,
    validate_matching_shapes,
    safe_divide,
    get_array_info,
)


def test_numeric_array_conversion():
    result = as_numeric_array([1, 2, 3])
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_reject_nonnumeric_input():
    with pytest.raises(TypeError):
        as_numeric_array(["hello", "world"])


def test_reject_complex_input():
    with pytest.raises(TypeError):
        as_numeric_array([1 + 2j, 3 + 4j])


def test_validate_required_dimensions():
    result = validate_array([1, 2, 3], ndim=1)
    assert result.ndim == 1


def test_reject_wrong_dimensions():
    with pytest.raises(ValueError):
        validate_array([[1, 2], [3, 4]], ndim=1)


def test_reject_empty_array_when_required():
    with pytest.raises(ValueError):
        validate_array([], allow_empty=False)


def test_reject_nan():
    with pytest.raises(ValueError):
        validate_array([1, np.nan], allow_nan=False)


def test_check_finite():
    np.testing.assert_array_equal(check_finite([1, 2, 3]), [1, 2, 3])


def test_check_finite_rejects_infinity():
    with pytest.raises(ValueError):
        check_finite([1, np.inf])


def test_matching_shapes():
    a, b = validate_matching_shapes([1, 2], [3, 4])
    assert a.shape == b.shape == (2,)


def test_mismatched_shapes():
    with pytest.raises(ValueError):
        validate_matching_shapes([1, 2], [3, 4, 5])


def test_safe_divide():
    np.testing.assert_allclose(
        safe_divide([10, 20], [2, 4]),
        [5, 5],
    )


def test_safe_divide_zero_raises():
    with pytest.raises(ZeroDivisionError):
        safe_divide([10, 20], [2, 0])


def test_safe_divide_zero_returns_nan():
    result = safe_divide([10, 20], [2, 0], zero_policy="nan")
    np.testing.assert_allclose(result, [5, np.nan], equal_nan=True)


def test_safe_divide_zero_returns_zero():
    result = safe_divide([10, 20], [2, 0], zero_policy="zero")
    np.testing.assert_array_equal(result, [5, 0])


def test_safe_divide_broadcasting():
    result = safe_divide([[10], [20]], [2, 4])
    np.testing.assert_allclose(result, [[5, 2.5], [10, 5]])


def test_array_metadata():
    info = get_array_info([[1, 2], [3, 4]])
    assert info["shape"] == (2, 2)
    assert info["ndim"] == 2
    assert info["size"] == 4
    assert info["dtype"] == "int64" or info["dtype"] == "int32"


def test_invalid_zero_policy():
    with pytest.raises(ValueError):
        safe_divide([1], [1], zero_policy="invalid")

def test_reject_boolean_input():
    with pytest.raises(TypeError):
        as_numeric_array([True, False])


def test_reject_complex_in_check_finite():
    with pytest.raises(TypeError):
        check_finite([1 + 2j])


def test_reject_infinity_when_disallowed():
    with pytest.raises(ValueError):
        validate_array([1, np.inf], allow_inf=False)


def test_reject_complex_safe_divide():
    with pytest.raises(TypeError):
        safe_divide([1 + 2j], [2])


def test_matching_shapes_requires_two_arrays():
    with pytest.raises(ValueError):
        validate_matching_shapes([1, 2])


def test_safe_divide_with_scalar_denominator():
    result = safe_divide([10, 20, 30], 10)
    np.testing.assert_allclose(result, [1, 2, 3])
