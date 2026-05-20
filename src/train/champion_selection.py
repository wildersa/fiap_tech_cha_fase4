from __future__ import annotations

from typing import Any


MIN_BASELINE_GAIN = 0.0
GAIN_TIE_MARGIN = 0.3


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _nested_metric(metrics: dict, section: str, key: str) -> float | None:
    section_value = metrics.get(section)
    if isinstance(section_value, dict):
        return _coerce_float(section_value.get(key))
    return None


def _flat_metric(metrics: dict, key: str) -> float | None:
    return _coerce_float(metrics.get(key))


def selection_lstm_mape(metrics: dict) -> float | None:
    for value in (
        _nested_metric(metrics, "lstm_test", "mape_pct"),
        _flat_metric(metrics, "test_lstm_mape_pct"),
        _nested_metric(metrics, "lstm_val", "mape_pct"),
        _flat_metric(metrics, "val_lstm_mape_pct"),
        _flat_metric(metrics, "lstm_mape_pct"),
    ):
        if value is not None and value > 0:
            return value
    return None


def baseline_mape(metrics: dict) -> float | None:
    for value in (
        _nested_metric(metrics, "baseline_test", "mape_pct"),
        _flat_metric(metrics, "test_baseline_mape_pct"),
        _flat_metric(metrics, "baseline_mape_pct"),
    ):
        if value is not None and value > 0:
            return value
    return None


def directional_accuracy(metrics: dict) -> float:
    for value in (
        _flat_metric(metrics, "directional_accuracy_test_lstm_pct"),
        _flat_metric(metrics, "directional_accuracy_pct"),
    ):
        if value is not None:
            return value
    return float("-inf")


def baseline_gain_pct(metrics: dict) -> float | None:
    for value in (
        _nested_metric(metrics, "relative_gain_vs_baseline_pct", "mape_pct"),
        _flat_metric(metrics, "baseline_gain_pct"),
        _flat_metric(metrics, "gain_mape_pct"),
    ):
        if value is not None:
            return value

    mape_lstm = selection_lstm_mape(metrics)
    mape_baseline = baseline_mape(metrics)
    if mape_lstm is not None and mape_baseline not in (None, 0):
        return ((mape_baseline - mape_lstm) / mape_baseline) * 100
    return None


def window_size(params: dict | None) -> int:
    params = params or {}
    return _coerce_int(params.get("window_size"), 60)


def inference_required_rows(params: dict | None) -> int:
    params = params or {}
    current_window_size = window_size(params)
    target_mode = params.get("target_mode", "log_returns")
    feature_mode = params.get("feature_mode", "single")
    if feature_mode == "single":
        return current_window_size + (1 if target_mode in {"log_returns", "returns"} else 0)
    return current_window_size + 21


def build_selection_record(metrics: dict | None, params: dict | None = None, run: Any | None = None) -> dict | None:
    metrics = metrics or {}
    mape_lstm = selection_lstm_mape(metrics)
    if mape_lstm is None:
        return None

    params = params or {}
    gain = baseline_gain_pct(metrics)
    mape_baseline = baseline_mape(metrics)
    if gain is None and mape_baseline not in (None, 0):
        gain = ((mape_baseline - mape_lstm) / mape_baseline) * 100

    current_window_size = window_size(params)
    record = {
        "run": run,
        "mape_lstm": mape_lstm,
        "mape": mape_lstm,
        "mape_baseline": mape_baseline,
        "baseline_gain_pct": gain,
        "gain_mape_pct": gain,
        "directional_accuracy": directional_accuracy(metrics),
        "inference_required_rows": inference_required_rows(params),
        "window_size": current_window_size,
    }
    return record


def _selection_sort_key(record: dict) -> tuple:
    directional = _coerce_float(record.get("directional_accuracy"), float("-inf"))
    mape_lstm = _coerce_float(record.get("mape_lstm", record.get("mape")), float("inf"))
    required_rows = _coerce_float(record.get("inference_required_rows"), float("inf"))
    current_window_size = _coerce_float(record.get("window_size"), float("inf"))
    return (-directional, mape_lstm, required_rows, current_window_size)


def select_best_record(records: list[dict] | None) -> dict | None:
    valid_records = [record for record in (records or []) if record is not None]
    eligible = []
    for record in valid_records:
        gain = _coerce_float(record.get("baseline_gain_pct"))
        if gain is not None and gain > MIN_BASELINE_GAIN:
            eligible.append((record, gain))

    if not eligible:
        return None

    max_gain = max(gain for _, gain in eligible)
    contenders = [record for record, gain in eligible if gain >= max_gain - GAIN_TIE_MARGIN]
    if not contenders:
        return None
    return sorted(contenders, key=_selection_sort_key)[0]


def should_promote_candidate(candidate: dict | None, incumbent: dict | None) -> bool:
    best = select_best_record([record for record in (incumbent, candidate) if record is not None])
    return best is candidate