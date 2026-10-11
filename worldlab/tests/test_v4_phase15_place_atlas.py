from __future__ import annotations

from pathlib import Path

import pytest

from worldlab_observation.place_atlas import (
    STATUS_MISSING,
    STATUS_SOURCE,
    STATUS_UNVERIFIED,
    TEMPORAL_CURRENT,
    TEMPORAL_HISTORICAL,
    TEMPORAL_STABLE_UNCONFIRMED,
    accessibility_row,
    field_status,
    haversine_km,
    identifiability_demo,
    m0_prediction_status,
    m1_prediction_status,
    match_grade,
    preference_invariance_when_attributes_match,
    reject_dual_use,
    snapshot_rows,
    statistical_category_row,
    synthetic_attribute_share,
    temporal_class_for,
)


SITE = {
    "group_id": "g",
    "site_uid": "stat_example",
    "official_name": "雪見遊憩區",
    "county": "苗栗縣",
}


def test_exact_name_and_county_is_source_reported_not_historical_description() -> None:
    assert match_grade("雪見遊憩區", "苗栗縣", "雪見遊憩區", "苗栗縣") == "exact_name_county"
    assert temporal_class_for("latitude", "exact_name_county") == TEMPORAL_STABLE_UNCONFIRMED
    assert temporal_class_for("attraction_description", "exact_name_county") == TEMPORAL_CURRENT
    assert field_status("24.42473", "exact_name_county") == STATUS_SOURCE


def test_fuzzy_name_is_not_written_as_a_place_coordinate() -> None:
    assert match_grade("觀霧遊客中心", "苗栗縣", "觀霧國家森林遊樂區", "苗栗縣") == "fuzzy_candidate"
    assert match_grade("汶水遊客中心", "苗栗縣", "汶水老街", "苗栗縣") == "fuzzy_candidate"
    assert match_grade("玉山管理處遊客服務中心", "南投縣", "玉山碾米廠", "南投縣") == ""
    rows = snapshot_rows(SITE, {"latitude": "24.5", "longitude": "121.1", "activity_tags": "16", "description": "x", "service_time": "", "fee_info": "", "facilities": "", "source_file": "x", "attraction_id": "a", "record_update": "2026-10-10"}, "fuzzy_candidate")
    assert all(row["status"] == STATUS_MISSING for row in rows)
    assert all(row["value"] == "" for row in rows)


def test_empty_fee_stays_missing() -> None:
    attraction = {
        "latitude": "24.17068",
        "longitude": "121.57261",
        "activity_tags": "1|2|7",
        "description": "步道持續封閉",
        "service_time": "",
        "fee_info": "",
        "facilities": "",
        "source_file": "AttractionList.xml",
        "attraction_id": "Attraction_376550000A_009392",
        "record_update": "2026-08-25T15:25:02+08:00",
    }
    rows = {row["field"]: row for row in snapshot_rows({**SITE, "official_name": "布洛灣遊憩區", "county": "花蓮縣"}, attraction, "exact_name_county")}
    assert rows["admission_or_cost_information"]["status"] == STATUS_MISSING
    assert rows["available_facilities"]["status"] == STATUS_MISSING
    assert rows["opening_or_access_conditions"]["status"] == STATUS_MISSING
    assert rows["attraction_description"]["temporal_class"] == TEMPORAL_CURRENT


def test_shared_statistical_category_does_not_create_an_m0_forecast() -> None:
    records = []
    for name in ("觀霧遊客中心", "雪見遊憩區", "汶水遊客中心"):
        site = {**SITE, "official_name": name, "site_uid": name}
        records.append(statistical_category_row(site, "國家公園", "2012", "2023"))
        records.extend(snapshot_rows(site, None, ""))
    m0 = m0_prediction_status(records)
    assert m0["prediction_status"] == "prediction_unavailable"
    assert m0["predicted_share"] is None
    assert m0["constant_utility_used"] is False
    m1 = m1_prediction_status(m0)
    assert m1["prediction_status"] == "prediction_unavailable"
    assert m1["skip_reason"] == "base_utility_missing"
    assert records[0]["temporal_class"] == TEMPORAL_HISTORICAL


def test_straight_line_distance_is_not_travel_time_and_fuzzy_distance_is_blank() -> None:
    distance = haversine_km(24.42473, 121.01340, 24.17068, 121.57261)
    assert distance > 50.0
    row = accessibility_row(SITE, "汶水老街", "苗栗縣", "fuzzy_candidate", "24.4", "121.0", "", "not_computed_for_fuzzy_candidate", "offline_credentials_absent", "名稱不同")
    assert row["distance_km"] == ""
    assert row["latitude"] == ""
    assert row["verification_status"] == STATUS_UNVERIFIED
    assert row["distance_kind"] == "not_computed_for_fuzzy_candidate"


def test_identical_attributes_make_preference_weights_drop_out() -> None:
    attributes = [[0.4, 0.2], [0.4, 0.2], [0.4, 0.2]]
    assert preference_invariance_when_attributes_match(attributes, [1.0, 0.0], [-2.0, 5.0])
    varied = synthetic_attribute_share([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]], [3.0, -1.0])
    assert abs(sum(varied) - 1.0) <= 1e-12


def test_same_popularity_cannot_be_attraction_and_crowding() -> None:
    with pytest.raises(ValueError, match="same popularity"):
        reject_dual_use("reported_visit_share_conditional", "reported_visit_share_conditional")


def test_weight_and_score_scale_are_not_separately_identified() -> None:
    demo = identifiability_demo()
    assert demo["weight_scale_tradeoff"] is True
    assert demo["identical_attributes_drop_preference"] is True


def test_runner_does_not_name_holdout_visit_files() -> None:
    source = Path("worldlab_observation/phase15_run.py").read_text(encoding="utf-8")
    assert "staging" not in source
    assert "v4_visits_" not in source
    assert "share" not in source.split("PANEL_COLUMNS")[1].split(")")[0]
