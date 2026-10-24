import numpy as np
from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    top_k,
    quick_select,
    binary_search,
)


def test_stable_sort():
    values = np.array([4, 2, 5, 1, 3])

    result = stable_sort(values)

    assert np.array_equal(result, [1, 2, 3, 4, 5])


def test_stable_sort_preserves_input():
    values = np.array([4, 2, 5, 1, 3])
    original = values.copy()

    stable_sort(values)

    assert np.array_equal(values, original)


def test_stable_sort_2d():
    values = np.array([[3, 1, 2], [6, 4, 5]])

    result = stable_sort(values)

    expected = np.array([[1, 2, 3], [4, 5, 6]])

    assert np.array_equal(result, expected)


def test_stable_sort_axis_zero():
    values = np.array([[3, 1], [1, 4], [2, 3]])

    result = stable_sort(values, axis=0)

    expected = np.array([[1, 1], [2, 3], [3, 4]])

    assert np.array_equal(result, expected)


def test_multi_key_sort_ascending():
    data = np.array([
        [2, 30],
        [1, 20],
        [2, 10],
        [1, 10],
    ])

    indices = multi_key_sort(data)

    expected = np.array([3, 1, 2, 0])

    assert np.array_equal(indices, expected)


def test_multi_key_sort_descending():
    data = np.array([
        [1, 10],
        [3, 30],
        [2, 20],
    ])

    indices = multi_key_sort(data, ascending=False)

    expected = np.array([0, 2, 1])

    assert np.array_equal(data[indices], [
        [1, 10],
        [2, 20],
        [3, 30],
    ]) == False

    assert np.array_equal(data[indices], [
        [3, 30],
        [2, 20],
        [1, 10],
    ])


def test_multi_key_sort_returns_indices():
    data = np.array([
        [3, 20],
        [1, 30],
        [2, 10],
    ])

    indices = multi_key_sort(data)

    assert np.array_equal(np.sort(indices), [0, 1, 2])


def test_multi_key_sort_duplicate_rows():
    data = np.array([
        [2, 10],
        [1, 20],
        [2, 10],
        [1, 10],
    ])

    indices = multi_key_sort(data)

    assert np.array_equal(data[indices], [
        [1, 10],
        [1, 20],
        [2, 10],
        [2, 10],
    ])


def test_multi_key_sort_input_unchanged():
    data = np.array([
        [3, 10],
        [1, 30],
        [2, 20],
    ])
    original = data.copy()

    multi_key_sort(data)

    assert np.array_equal(data, original)


def test_multi_key_sort_single_column():
    data = np.array([
        [3],
        [1],
        [2],
    ])

    indices = multi_key_sort(data)

    assert np.array_equal(data[indices], [
        [1],
        [2],
        [3],
    ])


def test_multi_key_sort_already_sorted():
    data = np.array([
        [1, 10],
        [1, 20],
        [2, 10],
        [2, 20],
    ])

    indices = multi_key_sort(data)

    assert np.array_equal(data[indices], data)


def test_multi_key_sort_reverse_sorted():
    data = np.array([
        [3, 30],
        [2, 20],
        [1, 10],
    ])

    indices = multi_key_sort(data)

    assert np.array_equal(data[indices], [
        [1, 10],
        [2, 20],
        [3, 30],
    ])


def test_top_k_largest_with_indices():
    values = np.array([10, 4, 7, 2, 9])

    indices = top_k(values, k=3)

    assert np.array_equal(np.sort(values[indices]), [7, 9, 10])


def test_top_k_smallest_with_indices():
    values = np.array([10, 4, 7, 2, 9])

    indices = top_k(values, k=2, largest=False)

    assert np.array_equal(np.sort(values[indices]), [2, 4])


def test_top_k_largest_without_indices():
    values = np.array([10, 4, 7, 2, 9])

    result = top_k(values, k=3, return_indices=False)

    assert np.array_equal(np.sort(result), [7, 9, 10])


def test_top_k_smallest_without_indices():
    values = np.array([10, 4, 7, 2, 9])

    result = top_k(
        values, k=2, largest=False, return_indices=False
    )

    assert np.array_equal(np.sort(result), [2, 4])


def test_top_k_all_elements():
    values = np.array([3, 1, 2])

    indices = top_k(values, k=3)

    assert np.array_equal(np.sort(values[indices]), [1, 2, 3])


def test_top_k_single_largest():
    values = np.array([5, 1, 8, 3])

    indices = top_k(values, k=1)

    assert values[indices[0]] == 8


def test_top_k_single_smallest():
    values = np.array([5, 1, 8, 3])

    result = top_k(
        values, k=1, largest=False, return_indices=False
    )

    assert result[0] == 1


def test_top_k_with_duplicate_values():
    values = np.array([5, 2, 5, 1, 2])

    result = top_k(values, k=2, return_indices=False)

    assert np.array_equal(np.sort(result), [5, 5])


def test_top_k_indices_are_valid():
    values = np.array([8, 3, 6, 1, 9])

    indices = top_k(values, k=3)

    assert indices.shape == (3,)
    assert np.all(indices >= 0)
    assert np.all(indices < values.size)
    assert np.unique(indices).size == 3


def test_top_k_preserves_input():
    values = np.array([8, 3, 6, 1, 9])
    original = values.copy()

    top_k(values, k=3)

    assert np.array_equal(values, original)


def test_top_k_rejects_zero_k():
    values = np.array([1, 2, 3])

    try:
        top_k(values, k=0)
        assert False, "Expected ValueError for k=0"
    except ValueError:
        pass


def test_top_k_rejects_k_too_large():
    values = np.array([1, 2, 3])

    try:
        top_k(values, k=4)
        assert False, "Expected ValueError when k exceeds array size"
    except ValueError:
        pass


def test_top_k_rejects_non_integer_k():
    values = np.array([1, 2, 3])

    try:
        top_k(values, k=2.5)
        assert False, "Expected TypeError for non-integer k"
    except TypeError:
        pass


def test_top_k_rejects_multidimensional_input():
    values = np.array([[1, 2], [3, 4]])

    try:
        top_k(values, k=2)
        assert False, "Expected ValueError for multidimensional input"
    except ValueError:
        pass


def test_top_k_rejects_invalid_largest():
    values = np.array([1, 2, 3])

    try:
        top_k(values, k=2, largest=1)
        assert False, "Expected TypeError for non-boolean largest"
    except TypeError:
        pass


def test_top_k_rejects_invalid_return_indices():
    values = np.array([1, 2, 3])

    try:
        top_k(values, k=2, return_indices=1)
        assert False, "Expected TypeError for non-boolean return_indices"
    except TypeError:
        pass


def test_quick_select_largest():
    values = np.array([7, 2, 9, 1, 5])

    index = quick_select(values, k=0)

    assert values[index] == 9


def test_quick_select_second_largest():
    values = np.array([7, 2, 9, 1, 5])

    index = quick_select(values, k=1)

    assert values[index] == 7


def test_quick_select_smallest():
    values = np.array([7, 2, 9, 1, 5])

    index = quick_select(values, k=0, largest=False)

    assert values[index] == 1


def test_quick_select_second_smallest():
    values = np.array([7, 2, 9, 1, 5])

    index = quick_select(values, k=1, largest=False)

    assert values[index] == 2


def test_quick_select_returns_value():
    values = np.array([7, 2, 9, 1, 5])

    result = quick_select(values, k=0, return_index=False)

    assert result == 9


def test_quick_select_returns_smallest_value():
    values = np.array([7, 2, 9, 1, 5])

    result = quick_select(
        values, k=0, largest=False, return_index=False
    )

    assert result == 1


def test_quick_select_last_largest_rank():
    values = np.array([7, 2, 9, 1, 5])

    index = quick_select(values, k=values.size - 1)

    assert values[index] == 1


def test_quick_select_single_element():
    values = np.array([42])

    index = quick_select(values, k=0)

    assert index == 0
    assert values[index] == 42


def test_quick_select_duplicate_values():
    values = np.array([4, 2, 2, 5, 2])

    index = quick_select(values, k=0, largest=False)

    assert values[index] == 2


def test_quick_select_preserves_input():
    values = np.array([7, 2, 9, 1, 5])
    original = values.copy()

    quick_select(values, k=1)

    assert np.array_equal(values, original)


def test_quick_select_rejects_multidimensional_input():
    values = np.array([[1, 2], [3, 4]])

    try:
        quick_select(values, k=0)
        assert False, "Expected ValueError for multidimensional input"
    except ValueError:
        pass


def test_quick_select_rejects_negative_k():
    values = np.array([1, 2, 3])

    try:
        quick_select(values, k=-1)
        assert False, "Expected ValueError for negative k"
    except ValueError:
        pass


def test_quick_select_rejects_k_out_of_range():
    values = np.array([1, 2, 3])

    try:
        quick_select(values, k=3)
        assert False, "Expected ValueError when k equals array size"
    except ValueError:
        pass


def test_quick_select_rejects_non_integer_k():
    values = np.array([1, 2, 3])

    try:
        quick_select(values, k=1.5)
        assert False, "Expected TypeError for non-integer k"
    except TypeError:
        pass


def test_quick_select_rejects_invalid_largest():
    values = np.array([1, 2, 3])

    try:
        quick_select(values, k=0, largest=1)
        assert False, "Expected TypeError for non-boolean largest"
    except TypeError:
        pass


def test_quick_select_rejects_invalid_return_index():
    values = np.array([1, 2, 3])

    try:
        quick_select(values, k=0, return_index=1)
        assert False, "Expected TypeError for non-boolean return_index"
    except TypeError:
        pass


def test_binary_search_empty_array():
    values = np.array([], dtype=int)

    index, found = binary_search(values, 5)

    assert index == 0
    assert found is False


def test_binary_search_single_element_found():
    values = np.array([7])

    index, found = binary_search(values, 7)

    assert index == 0
    assert found is True


def test_binary_search_single_element_missing():
    values = np.array([7])

    index, found = binary_search(values, 3)

    assert index == 0
    assert found is False


def test_binary_search_first_element():
    values = np.array([2, 4, 6, 8, 10])

    index, found = binary_search(values, 2)

    assert index == 0
    assert found is True


def test_binary_search_last_element():
    values = np.array([2, 4, 6, 8, 10])

    index, found = binary_search(values, 10)

    assert index == 4
    assert found is True


def test_binary_search_target_smaller_than_all():
    values = np.array([2, 4, 6, 8, 10])

    index, found = binary_search(values, 1)

    assert index == 0
    assert found is False


def test_binary_search_target_larger_than_all():
    values = np.array([2, 4, 6, 8, 10])

    index, found = binary_search(values, 12)

    assert index == 5
    assert found is False


def test_binary_search_insertion_between_elements():
    values = np.array([10, 20, 30, 40, 50])

    index, found = binary_search(values, 35)

    assert index == 3
    assert found is False

if __name__ == "__main__":
    test_stable_sort()
    test_stable_sort_preserves_input()
    test_stable_sort_2d()
    test_stable_sort_axis_zero()
    test_multi_key_sort_ascending()
    test_multi_key_sort_returns_indices()
    test_multi_key_sort_descending()
    test_multi_key_sort_duplicate_rows()
    test_multi_key_sort_input_unchanged()
    test_multi_key_sort_single_column()
    test_multi_key_sort_already_sorted()
    test_multi_key_sort_reverse_sorted()

    test_top_k_largest_with_indices()
    test_top_k_smallest_with_indices()
    test_top_k_largest_without_indices()
    test_top_k_smallest_without_indices()
    test_top_k_all_elements()
    test_top_k_single_largest()
    test_top_k_single_smallest()
    test_top_k_with_duplicate_values()
    test_top_k_indices_are_valid()
    test_top_k_preserves_input()
    test_top_k_rejects_zero_k()
    test_top_k_rejects_k_too_large()
    test_top_k_rejects_non_integer_k()
    test_top_k_rejects_multidimensional_input()
    test_top_k_rejects_invalid_largest()
    test_top_k_rejects_invalid_return_indices()

    test_quick_select_largest()
    test_quick_select_second_largest()
    test_quick_select_smallest()
    test_quick_select_second_smallest()
    test_quick_select_returns_value()
    test_quick_select_returns_smallest_value()
    test_quick_select_last_largest_rank()
    test_quick_select_single_element()
    test_quick_select_duplicate_values()
    test_quick_select_preserves_input()
    test_quick_select_rejects_multidimensional_input()
    test_quick_select_rejects_negative_k()
    test_quick_select_rejects_k_out_of_range()
    test_quick_select_rejects_non_integer_k()
    test_quick_select_rejects_invalid_largest()
    test_quick_select_rejects_invalid_return_index()

    test_binary_search_empty_array()
    test_binary_search_single_element_found()
    test_binary_search_single_element_missing()
    test_binary_search_first_element()
    test_binary_search_last_element()
    test_binary_search_target_smaller_than_all()
    test_binary_search_target_larger_than_all()
    test_binary_search_insertion_between_elements()

    print("All 20 test cases passed!")