from __future__ import annotations

from worldlab_data.phase12 import (
    alignment_pattern,
    classify_category_discrepancy,
    confirmed_parser_error,
    exclude_holdout_rows,
    link_decision,
    strict_asof_eligible,
    valid_share_vector,
)


def test_category_discrepancy_not_auto_corrected() -> None:
    reason = classify_category_discrepancy("宗教場所", "國家級風景特定區", "", filled_by_parser=False)
    assert reason == "unresolved"
    assert reason != "parser_error"
    carried = classify_category_discrepancy("國家級風景特定區", "宗教場所", "", filled_by_parser=True)
    assert carried == "parser_error"


def test_cross_type_name_is_not_a_safe_link() -> None:
    blocked = link_decision(True, True, True, False, True, True)
    assert blocked["safe_to_link"] is False
    assert blocked["identity_status"] == "rule_inferred_not_confirmed"
    missing = link_decision(False, True, True, False, True, False)
    assert missing["safe_to_link"] is False
    assert missing["identity_status"] == "not_found"
    assert "不補值" in str(missing["continuity_evidence"])


def test_same_category_rule_link() -> None:
    linked = link_decision(True, True, True, True, True, False)
    assert linked["safe_to_link"] is True
    assert linked["identity_status"] == "rule_inferred"
    confirmed = link_decision(True, True, True, True, True, True)
    assert confirmed["identity_status"] == "official_source_confirmed"


def test_alignment_equality_is_not_parser_proof() -> None:
    pattern = alignment_pattern(11939.0, 4175.0, 11939.0, 53180.0)
    assert pattern == "equals_previous_visit_row"
    assert confirmed_parser_error(4175.0, 4175.0) is False
    assert confirmed_parser_error(4175.0, 11939.0) is True


def test_forecast_share_vector_and_holdout() -> None:
    assert valid_share_vector([0.2, 0.8]) is True
    assert valid_share_vector([0.2, 0.7]) is False
    assert valid_share_vector([-0.1, 1.1]) is False
    assert strict_asof_eligible("") is False
    assert strict_asof_eligible("2024-03-15") is True
    kept, held = exclude_holdout_rows(
        [{"year_month": "2023-12", "share": 1.0}, {"year_month": "2025-01", "share": 1.0}]
    )
    assert [row["year_month"] for row in kept] == ["2023-12"]
    assert held == ["2025-01"]
