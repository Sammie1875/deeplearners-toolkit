
import numpy as np
import pytest

from numcompute.rank import rankdata, percentile_rank


def test_average_ranking_with_ties():
    result = rankdata([40, 10, 30, 10, 50])
    np.testing.assert_allclose(
        result, [4, 1.5, 3, 1.5, 5]
    )


def test_dense_ranking():
    result = rankdata([40, 10, 30, 10, 50], method="dense")
    np.testing.assert_array_equal(
        result, [3, 1, 2, 1, 4]
    )


def test_ordinal_ranking():
    result = rankdata([40, 10, 30, 10, 50], method="ordinal")
    np.testing.assert_array_equal(
        result, [4, 1, 3, 2, 5]
    )


def test_descending_ranking():
    result = rankdata([10, 20, 20, 40], ascending=False)
    np.testing.assert_allclose(result, [4, 2.5, 2.5, 1])


def test_empty_input():
    result = rankdata([])
    assert result.size == 0


def test_single_value():
    np.testing.assert_array_equal(rankdata([42]), [1])


def test_nan_omit():
    result = rankdata([10, np.nan, 30], nan_policy="omit")
    np.testing.assert_allclose(
        result, [1, np.nan, 2], equal_nan=True
    )


def test_nan_raise():
    with pytest.raises(ValueError):
        rankdata([10, np.nan, 30])


def test_nan_propagate():
    result = rankdata([10, np.nan, 30], nan_policy="propagate")
    assert np.isnan(result).all()


def test_invalid_method():
    with pytest.raises(ValueError):
        rankdata([1, 2, 3], method="unknown")


def test_multidimensional_axis():
    values = np.array([[30, 10], [20, 40]])
    result = rankdata(values, axis=1)
    np.testing.assert_array_equal(result, [[2, 1], [1, 2]])


def test_percentile_rank():
    assert percentile_rank([10, 20, 30, 40, 50], 40) == 80.0


def test_percentile_rank_with_ties():
    assert percentile_rank([10, 20, 20, 40], 20) == 62.5


def test_percentile_rank_empty_data():
    with pytest.raises(ValueError):
        percentile_rank([], 10)


def test_percentile_rank_weak_and_strict():
    values = [10, 20, 20, 40]
    assert percentile_rank(values, 20, kind="weak") == 75.0
    assert percentile_rank(values, 20, kind="strict") == 25.0


def test_invalid_input_dimensions():
    with pytest.raises(ValueError):
        rankdata([[1, 2], [3, 4]], axis=3)


def test_min_and_max_ranking():
    values = [10, 20, 20, 40]
    np.testing.assert_array_equal(
        rankdata(values, method="min"), [1, 2, 2, 4]
    )
    np.testing.assert_array_equal(
        rankdata(values, method="max"), [1, 3, 3, 4]
    )

def test_descending_ordinal_preserves_tie_order():
    result = rankdata(
        [20, 10, 20, 30],
        method="ordinal",
        ascending=False,
    )
    np.testing.assert_array_equal(result, [2, 4, 3, 1])


def test_percentile_rank_multiple_scores():
    result = percentile_rank([10, 20, 30, 40], [10, 25, 40])
    np.testing.assert_allclose(result, [25, 50, 100])


def test_percentile_rank_nan_score():
    result = percentile_rank([10, 20, 30], np.nan)
    assert np.isnan(result)


def test_nan_omit_all_missing():
    with pytest.raises(ValueError):
        percentile_rank([np.nan, np.nan], 10, nan_policy="omit")


def test_negative_axis():
    values = np.array([[30, 10], [20, 40]])
    result = rankdata(values, axis=-1)
    np.testing.assert_array_equal(result, [[2, 1], [1, 2]])


def test_invalid_nan_policy():
    with pytest.raises(ValueError):
        rankdata([1, 2, 3], nan_policy="invalid")


def test_large_integer_values_remain_distinct():
    values = [2**53, 2**53 + 1]
    np.testing.assert_array_equal(rankdata(values), [1, 2])


def test_infinity_ranking():
    result = rankdata([-np.inf, 5, np.inf])
    np.testing.assert_array_equal(result, [1, 2, 3])

def test_descending_ordinal_preserves_tie_order():
    result = rankdata(
        [20, 10, 20, 30],
        method="ordinal",
        ascending=False,
    )
    np.testing.assert_array_equal(result, [2, 4, 3, 1])


def test_percentile_rank_multiple_scores():
    result = percentile_rank([10, 20, 30, 40], [10, 25, 40])
    np.testing.assert_allclose(result, [25, 50, 100])


def test_percentile_rank_nan_score():
    result = percentile_rank([10, 20, 30], np.nan)
    assert np.isnan(result)


def test_nan_omit_all_missing():
    with pytest.raises(ValueError):
        percentile_rank([np.nan, np.nan], 10, nan_policy="omit")


def test_negative_axis():
    values = np.array([[30, 10], [20, 40]])
    result = rankdata(values, axis=-1)
    np.testing.assert_array_equal(result, [[2, 1], [1, 2]])


def test_invalid_nan_policy():
    with pytest.raises(ValueError):
        rankdata([1, 2, 3], nan_policy="invalid")


def test_large_integer_values_remain_distinct():
    values = [2**53, 2**53 + 1]
    np.testing.assert_array_equal(rankdata(values), [1, 2])
def test_infinity_ranking():
    result = rankdata([-np.inf, 5, np.inf])
    np.testing.assert_array_equal(result, [1, 2, 3])
