from __future__ import annotations

from worldlab_data.phase11 import (
    assess_group_month,
    concentration_metrics,
    exclude_holdout,
    hypothetical_scaled_hhi,
    identity_semantics,
    leader_turnover,
    share_total_variation,
)


def test_share_sum() -> None:
    assessed = assess_group_month({"a": 1.0, "b": 3.0}, source_conflict=False)
    assert assessed["share_defined"] is True
    shares = assessed["shares"]
    assert abs(sum(shares.values()) - 1.0) < 1e-12
    assert shares["b"] == 0.75


def test_all_zero_group_month() -> None:
    assessed = assess_group_month({"a": 0.0, "b": 0.0}, source_conflict=False)
    assert assessed["share_defined"] is False
    assert assessed["exclusion_reason"] == "all_zero_group_total"
    assert assessed["group_total_visits"] == 0.0
    assert assessed["raw_hhi"] is None
    assert assessed["normalized_hhi"] is None
    assert assessed["raw_hhi"] != 0


def test_share_defined_month_count() -> None:
    months = [
        assess_group_month({"a": 2.0, "b": 2.0}, source_conflict=False),
        assess_group_month({"a": 0.0, "b": 0.0}, source_conflict=False),
        assess_group_month({"a": 5.0, "b": 1.0}, source_conflict=True),
        assess_group_month({"a": 4.0, "b": 1.0}, source_conflict=False),
    ]
    assert sum(row["reported_months"] for row in months) == 3
    assert sum(row["positive_total_months"] for row in months) == 2
    assert sum(row["share_defined"] for row in months) == 2
    assert months[1]["exclusion_reason"] == "all_zero_group_total"
    assert months[2]["exclusion_reason"] == "source_conflict"
    assert months[2]["group_total_visits"] == 6.0


def test_hhi_normalization() -> None:
    equal = concentration_metrics({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
    monopoly = concentration_metrics({"a": 5.0, "b": 0.0, "c": 0.0})
    assert abs(float(equal["raw_hhi"]) - 0.25) < 1e-12
    assert abs(float(equal["normalized_hhi"]) - 0.0) < 1e-12
    assert abs(float(equal["effective_number_of_sites"]) - 4.0) < 1e-12
    assert abs(float(monopoly["raw_hhi"]) - 1.0) < 1e-12
    assert abs(float(monopoly["normalized_hhi"]) - 1.0) < 1e-12


def test_monthly_tv() -> None:
    distance = share_total_variation({"a": 0.7, "b": 0.3}, {"a": 0.4, "b": 0.6})
    assert distance is not None
    assert abs(distance - 0.3) < 1e-12
    assert share_total_variation({"a": 1.0}, {"a": 0.5, "b": 0.5}) is None


def test_leader_tie_handling() -> None:
    tied = concentration_metrics({"a": 2.0, "b": 2.0})
    assert tied["leader_tie"] is True
    assert tied["leading_site_uid"] == ""
    assert leader_turnover("a", False, "", True) is None
    assert leader_turnover("a", False, "b", False) is True
    assert leader_turnover("a", False, "a", False) is False


def test_identity_confidence_semantics() -> None:
    fields = identity_semantics(["provisional_rule_match", "singleton"])
    assert fields["rule_match_coverage"] == 1.0
    assert fields["externally_verified_identity_status"] == "unknown"
    assert "identity_confidence" not in fields
    scaled = hypothetical_scaled_hhi({"a": 8.0, "b": 2.0}, "a", 1.2)
    assert scaled is not None
    assert scaled != concentration_metrics({"a": 8.0, "b": 2.0})["normalized_hhi"]


def test_no_2025_holdout_leakage() -> None:
    kept, held = exclude_holdout(["2024-12", "2025-01", "2026-03"])
    assert kept == ["2024-12"]
    assert held == ["2025-01", "2026-03"]
    assert all(int(month[:4]) < 2025 for month in kept)
