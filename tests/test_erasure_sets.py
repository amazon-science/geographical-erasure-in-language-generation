from numpy import testing

from geographical_erasure import get_erasure_set


def test_get_erasure_set():
    ground_truth_valid = {"A": 0.5, "B": 0.3, "C": 0.2}
    ground_truth_invalid = {"A": 0.2, "B": 0.3, "C": 0.2}

    pred_valid = {
        "B": 0.1,
        "C": 0.7,
        "A": 0.2,
    }  # check †hat everything also works in keys are permuted
    pred_invalid = {
        "B": 0.1,
        "C": 0.2,
        "A": 0.2,
    }  # check †hat everything also works in keys are permuted

    # test that errors are being thrown correctly
    testing.assert_raises(
        AssertionError, get_erasure_set, pred_invalid, ground_truth_valid, 2
    )
    testing.assert_raises(
        AssertionError, get_erasure_set, pred_valid, ground_truth_invalid, 2
    )

    erasure_set_r2 = get_erasure_set(pred_valid, ground_truth_valid, 2)
    testing.assert_equal(erasure_set_r2, {"A": 0.2, "B": 0.1})

    erasure_set_r100 = get_erasure_set(pred_valid, ground_truth_valid, 100)
    testing.assert_equal(erasure_set_r100, {})
