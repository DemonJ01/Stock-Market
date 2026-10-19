import numpy as np
import pytest
from Main import make_sequences, safe_mape


def test_rolling_windows_are_chronological():
    x, y = make_sequences(np.array([1, 2, 3, 4, 5]), 2)
    assert x.shape == (3, 2, 1)
    assert y.tolist() == [3.0, 4.0, 5.0]
    assert x[0, :, 0].tolist() == [1.0, 2.0]


def test_short_history_returns_empty_sequences():
    x, y = make_sequences([1, 2], 2)
    assert x.shape == (0, 2, 1)
    assert y.size == 0


def test_mape_excludes_zero_actuals():
    assert safe_mape([100, 0, 200], [90, 30, 180]) == pytest.approx(10.0)


def test_mape_rejects_mismatched_series():
    with pytest.raises(ValueError):
        safe_mape([1, 2], [1])


def test_mape_ignores_nonfinite_points():
    assert safe_mape([100.0, float("nan"), 200.0], [90.0, 50.0, 180.0]) == pytest.approx(10.0)


def test_mape_all_zero_actual_values():
    assert np.isnan(safe_mape([0, 0], [1, 5]))


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        make_sequences([1, 2, 3], 0)
