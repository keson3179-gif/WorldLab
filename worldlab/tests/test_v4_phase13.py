from __future__ import annotations

from pathlib import Path

from worldlab_data.phase13 import (
    EVALUATION_MONTHS,
    TRAIN_MONTHS,
    WEIGHTS_TUNED_ON_EVALUATION,
    classify_priority_identity,
    common_scoring_months,
    evaluable_months,
    independent_sample_size,
    information_cutoff,
    is_defined_share_month,
    leakage_failures,
    nonoverlapping_blocks,
    predict_shares,
    reject_holdout_months,
    rolling_origins,
    score_share_forecast,
    shift_month,
)
from worldlab_data.phase12 import valid_share_vector


def _series(start: str, count: int, shares: dict[str, float]) -> dict[str, dict[str, float]]:
    history = {}
    month = start
    for _ in range(count):
        history[month] = dict(shares)
        month = shift_month(month, 1)
    return history


def test_share_prediction_simplex() -> None:
    membership = ("a", "b", "c")
    history = _series("2012-01", 36, {"a": 0.5, "b": 0.3, "c": 0.2})
    target = "2014-06"
    for scenario in ("ex_post_max_information", "assumed_publication_delay"):
        for model in (
            "persistence",
            "same_month_last_year",
            "seasonal_month_mean",
            "trailing_3_mean",
            "trailing_12_mean",
            "fixed_equal_weight",
            "uniform_reference",
        ):
            predicted = predict_shares(history, membership, target, scenario, model)
            assert predicted["status"] == "predicted"
            shares = predicted["shares"]
            assert valid_share_vector([shares[uid] for uid in membership])
            assert abs(sum(shares.values()) - 1.0) <= 1e-8
            assert WEIGHTS_TUNED_ON_EVALUATION is False


def test_rolling_origin_no_leakage() -> None:
    membership = ("a", "b")
    history = _series("2012-01", 48, {"a": 0.8, "b": 0.2})
    target = "2014-01"
    history[target] = {"a": 0.1, "b": 0.9}
    history[shift_month(target, 1)] = {"a": 0.0, "b": 1.0}
    predicted = predict_shares(history, membership, target, "ex_post_max_information", "persistence")
    assert predicted["shares"] == {"a": 0.8, "b": 0.2}
    assert leakage_failures(predicted, target) == []
    assert target not in predicted["source_months"]
    origins = rolling_origins(list(history))
    assert origins
    assert all(origin > "2013-01" for origin in origins)


def test_asof_lag_one() -> None:
    membership = ("a", "b")
    history = {}
    month = "2012-01"
    for index in range(40):
        history[month] = {"a": 0.2 + (index % 5) * 0.1, "b": 0.8 - (index % 5) * 0.1}
        month = shift_month(month, 1)
    target = "2014-05"
    predicted = predict_shares(history, membership, target, "ex_post_max_information", "persistence")
    cutoff = information_cutoff(target, "ex_post_max_information")
    assert cutoff == shift_month(target, -1)
    assert predicted["source_months"] == [cutoff]
    trailing = predict_shares(history, membership, target, "ex_post_max_information", "trailing_3_mean")
    assert trailing["source_months"] == [shift_month(target, -3), shift_month(target, -2), cutoff]
    assert leakage_failures(trailing, target) == []


def test_asof_lag_three() -> None:
    membership = ("a", "b")
    history = {}
    month = "2012-01"
    for index in range(40):
        history[month] = {"a": 0.2 + (index % 5) * 0.1, "b": 0.8 - (index % 5) * 0.1}
        month = shift_month(month, 1)
    target = "2014-05"
    cutoff = information_cutoff(target, "assumed_publication_delay")
    assert cutoff == shift_month(target, -3)
    predicted = predict_shares(history, membership, target, "assumed_publication_delay", "persistence")
    assert predicted["source_months"] == [cutoff]
    assert predicted["shares"] != predict_shares(history, membership, target, "ex_post_max_information", "persistence")["shares"]
    trailing = predict_shares(history, membership, target, "assumed_publication_delay", "trailing_3_mean")
    assert trailing["source_months"] == [shift_month(target, -5), shift_month(target, -4), cutoff]
    assert shift_month(target, -1) not in trailing["source_months"]
    assert leakage_failures(predicted, target) == []


def test_missing_month_no_imputation() -> None:
    membership = ("a", "b")
    history = _series("2013-01", 6, {"a": 0.7, "b": 0.3})
    del history["2013-05"]
    predicted = predict_shares(history, membership, "2013-06", "ex_post_max_information", "trailing_3_mean")
    assert predicted["status"] == "prediction_unavailable"
    assert predicted["skip_reason"] == "trailing_window_incomplete"
    assert predicted["shares"] is None
    uniform = predict_shares(history, membership, "2013-06", "ex_post_max_information", "uniform_reference")
    assert predicted["shares"] != uniform["shares"]


def test_all_zero_month_exclusion() -> None:
    membership = ("a", "b")
    assert is_defined_share_month({"a": 0.0, "b": 0.0}, membership) is False
    history = _series("2012-01", 30, {"a": 0.6, "b": 0.4})
    history["2014-01"] = {"a": 0.0, "b": 0.0}
    defined = [month for month, shares in history.items() if is_defined_share_month(shares, membership)]
    assert "2014-01" not in defined
    assert "2014-01" not in evaluable_months(defined)
    predicted = predict_shares(history, membership, "2014-02", "ex_post_max_information", "persistence")
    assert predicted["status"] == "prediction_unavailable"
    assert predicted["shares"] is None


def test_fixed_group_membership() -> None:
    membership = ("a", "b")
    history = _series("2012-01", 30, {"a": 0.6, "b": 0.4})
    history["2014-03"] = {"a": 0.6, "b": 0.4, "c": 0.0}
    assert is_defined_share_month(history["2014-03"], membership) is False
    predicted = predict_shares(history, membership, "2014-04", "ex_post_max_information", "persistence")
    assert predicted["status"] == "prediction_unavailable"
    intact = predict_shares(
        {month: shares for month, shares in history.items() if month != "2014-03"},
        membership,
        "2014-02",
        "ex_post_max_information",
        "persistence",
    )
    assert set(intact["shares"]) == set(membership)


def test_same_scoring_calendar() -> None:
    available = {
        "persistence": {"2014-01", "2014-02", "2014-03"},
        "seasonal_month_mean": {"2014-02", "2014-03", "2014-04"},
        "uniform_reference": {"2014-01", "2014-02", "2014-03", "2014-04"},
    }
    shared = common_scoring_months(available)
    assert shared == ["2014-02", "2014-03"]
    assert all(month in available[model] for model in available for month in shared)


def test_no_2025_data_access() -> None:
    kept, held = reject_holdout_months(["2024-12", "2025-01", "2026-03"])
    assert kept == ["2024-12"]
    assert held == ["2025-01", "2026-03"]
    source = (Path(__file__).resolve().parents[1] / "worldlab_data" / "phase13_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source


def test_overlapping_windows_not_independent() -> None:
    defined = []
    month = "2012-01"
    for _ in range(120):
        defined.append(month)
        month = shift_month(month, 1)
    origins = rolling_origins(defined, TRAIN_MONTHS, EVALUATION_MONTHS)
    selected = evaluable_months(defined)
    assert len(origins) == 85
    assert len(selected) == 96
    assert independent_sample_size(selected) == 96
    assert independent_sample_size(selected) != len(origins) * EVALUATION_MONTHS
    blocks = nonoverlapping_blocks(selected)
    assert len(blocks) == 8
    kinmen = []
    month = "2018-01"
    for _ in range(41):
        kinmen.append(month)
        month = shift_month(month, 1)
    assert len(rolling_origins(kinmen)) == 6
    assert len(evaluable_months(kinmen)) == 17


def test_2024_identity_evidence_separation() -> None:
    reviewed = classify_priority_identity(True, False, "電子計數器", "電子計數器", 1, True)
    assert reviewed["review_status"] == "cross_source_evidence_candidate"
    assert reviewed["confirmed_manual"] is False
    assert reviewed["physical_or_statistical_identity_continuity"] != reviewed["measurement_continuity"]
    assert reviewed["measurement_continuity"] == "not_established"
    assert reviewed["baseline_sample_changed"] is False
    assert reviewed["same_year_statistical_correspondence"] is True
    scored = score_share_forecast({"a": 0.25, "b": 0.75}, {"a": 0.25, "b": 0.75})
    assert scored["share_vector_TV"] == 0.0
    assert scored["status"] == "scored"
