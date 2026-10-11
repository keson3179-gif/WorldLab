from __future__ import annotations

from pathlib import Path

import pytest

from worldlab_observation.features import (
    CURRENT_SNAPSHOT,
    DECISION_NO_GO,
    EX_POST_WEATHER,
    HISTORICAL_ASOF,
    RETROSPECTIVE_PROXY,
    SOCIAL_UNOBSERVED,
    allows_strict_cutoff,
    common_feature_cancels,
    constant_shift_cancels,
    decision_from_eligibility,
    has_site_variation,
    has_temporal_variation,
    identity_blocks_link,
    refuse_visit_share_as_social,
    reject_imputed_missing,
)


def test_feature_provenance() -> None:
    row = {
        "feature_name": "weekend_days",
        "source_file": "gregorian_calendar",
        "historical_availability_status": HISTORICAL_ASOF,
        "source_date": "",
    }
    assert row["feature_name"]
    assert row["source_file"]
    assert row["historical_availability_status"] in {HISTORICAL_ASOF, RETROSPECTIVE_PROXY, CURRENT_SNAPSHOT}


def test_historical_availability() -> None:
    assert allows_strict_cutoff(HISTORICAL_ASOF, "none") is True
    assert allows_strict_cutoff(RETROSPECTIVE_PROXY, "publication_date_unknown") is False
    assert allows_strict_cutoff(CURRENT_SNAPSHOT, "none") is False
    assert allows_strict_cutoff(HISTORICAL_ASOF, EX_POST_WEATHER) is False


def test_group_common_feature_cancellation() -> None:
    assert common_feature_cancels([8.0, 8.0, 8.0]) is True
    assert constant_shift_cancels([0.2, 1.0, -0.4], 3.0) is True
    assert has_site_variation([8.0, 8.0, 8.0]) is False


def test_site_month_variation() -> None:
    assert has_site_variation([12.0, 18.0, 15.0]) is True
    assert has_site_variation([12.0]) is False
    assert has_temporal_variation([10.0, 11.0, 10.0]) is True
    assert has_temporal_variation([10.0, 10.0]) is False


def test_future_weather_leakage() -> None:
    assert allows_strict_cutoff(RETROSPECTIVE_PROXY, EX_POST_WEATHER) is False
    assert EX_POST_WEATHER == "ex_post_conditional_analysis"


def test_source_independence() -> None:
    with pytest.raises(ValueError, match="social exposure"):
        refuse_visit_share_as_social("reported_visit_share")
    refuse_visit_share_as_social("station_month_heat_metrics_v0.2.csv")
    assert SOCIAL_UNOBSERVED == "external_social_information_unobserved"


def test_missing_not_imputed() -> None:
    reject_imputed_missing(False, False)
    with pytest.raises(ValueError, match="imputed"):
        reject_imputed_missing(True, True)


def test_identity_unresolved_gate() -> None:
    assert identity_blocks_link(False, False) is True
    assert identity_blocks_link(True, False) is True
    assert identity_blocks_link(True, True) is False


def test_no_2025_holdout_access() -> None:
    source = Path("worldlab_observation/phase17_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source
    assert "share" not in source.split("PANEL_COLUMNS")[1].split(")")[0]
    assert "2025" not in source.split("READ_PATHS")[1].split(")")[0]


def test_decision_requires_within_group_contrast() -> None:
    rows = [
        {"share_increment": "false", "historical_availability_status": HISTORICAL_ASOF, "analysis_role": "not_a_share_model"},
        {"share_increment": "false", "historical_availability_status": RETROSPECTIVE_PROXY, "analysis_role": EX_POST_WEATHER},
    ]
    assert decision_from_eligibility(rows) == DECISION_NO_GO
