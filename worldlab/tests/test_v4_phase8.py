from __future__ import annotations

import pandas as pd

from worldlab_data.phase8 import (
    classify_annual_difference,
    confirmation_decision,
    geographic_groups,
    parse_annual_matrix,
    substitution_candidates,
)


def test_only_exact_unique_is_confirmed() -> None:
    exact = confirmation_decision(
        [
            {
                "match_method": "exact_unique_same_county",
                "candidate_poi_id": "poi",
                "official_name": "陽明書屋",
                "candidate_name": "陽明書屋",
                "same_key": True,
                "candidate_latitude": "25.1",
                "candidate_longitude": "121.5",
            }
        ]
    )
    fuzzy = confirmation_decision(
        [
            {
                "match_method": "fuzzy_candidate",
                "candidate_poi_id": "poi",
                "official_name": "白沙灣",
                "candidate_name": "白沙灣海水浴場",
                "same_key": False,
                "candidate_latitude": "25.2",
                "candidate_longitude": "121.5",
            }
        ]
    )
    multiple = confirmation_decision(
        [
            {
                "match_method": "exact_name_multiple_pois",
                "candidate_poi_id": "a",
                "same_key": True,
                "candidate_latitude": "1",
                "candidate_longitude": "2",
            },
            {
                "match_method": "exact_name_multiple_pois",
                "candidate_poi_id": "b",
                "same_key": True,
                "candidate_latitude": "1.1",
                "candidate_longitude": "2.1",
            },
        ]
    )
    assert exact[0] == "confirmed_manual"
    assert fuzzy[0] == "needs_manual_review"
    assert multiple[0] == "needs_manual_review"


def test_annual_class_does_not_rebuild_months() -> None:
    assert classify_annual_difference(["reported"] * 11 + ["missing"], 10, 12, "absent", None, None) == "missing_month"
    assert classify_annual_difference(["reported"] * 12, 10, None, "absent", None, None) == "format_parse"
    assert classify_annual_difference(["reported"] * 12, 120, 100, "present", 120, 100) == "unknown"
    assert classify_annual_difference(["reported"] * 12, 120, 100, "present", 90, 90) == "version_difference"
    assert classify_annual_difference(["reported"] * 12, 120, 100, "absent", None, None) == "unknown"


def test_excel_parent_row_is_not_a_visit() -> None:
    header = ["類型", "觀光遊憩區", "縣市", "100年\n1月", "100年\n2月", "100年\n3月", "100年\n4月", "100年\n5月", "100年\n6月", "100年\n1月-12月"]
    parent = ["國家公園", "陽明山國家公園", "", "", "", "", "", "", "", ""]
    child = ["國家公園", "陽明書屋", "臺北市", 1, 1, 1, 1, 1, 1, 9]
    parsed = parse_annual_matrix([header, parent, child], "sample.xlsx")
    assert {row["official_name"] for row in parsed} == {"陽明書屋"}
    assert {row["parent_area"] for row in parsed} == {"陽明山國家公園"}
    assert sum(row["visits"] for row in parsed) == 6
    assert {row["year"] for row in parsed} == {2011}


def test_prior_year_columns_are_not_added_into_the_current_year() -> None:
    header = [
        "類型",
        "觀光遊憩區",
        "縣市",
        "100年\n1月",
        "100年\n2月",
        "100年\n3月",
        "100年\n4月",
        "100年\n5月",
        "100年\n6月",
        "100年\n1月-12月",
        "99年\n1月",
        "99年\n2月",
        "99年\n3月",
        "99年\n4月",
        "99年\n5月",
        "99年\n6月",
        "99年\n1月-12月",
        "遊客人次計算方式",
    ]
    child = ["國家公園", "陽明書屋", "臺北市", 1, 1, 1, 1, 1, 1, 9, 8, 8, 8, 8, 8, 8, 80, "電子計數器"]
    parsed = parse_annual_matrix([header, child], "100.xlsx")
    assert {row["year"] for row in parsed} == {2011}
    assert sum(row["visits"] for row in parsed) == 6
    assert {row["measurement_method"] for row in parsed} == {"電子計數器"}


def test_group_size_limit_is_removed_and_park_is_not_substitution() -> None:
    history = pd.DataFrame(
        [
            {"site_id": f"s{index}", "official_name": f"步道{index}", "county": "臺北市", "parent_area": "陽明山國家公園"}
            for index in range(9)
        ]
        + [
            {"site_id": "admin", "official_name": "陽明山遊客中心", "county": "臺北市", "parent_area": "陽明山國家公園"},
            {"site_id": "dash", "official_name": "九族文化村", "county": "南投縣", "parent_area": "-"},
        ]
    )
    geographic = geographic_groups(history)
    assert geographic["group_id"].nunique() == 1
    assert str(geographic["group_id"].iloc[0]).startswith("geo_")
    assert geographic["group_size"].iloc[0] == 10
    assert "dash" not in set(geographic["site_id"])
    substitution = substitution_candidates(geographic)
    assert "陽明山遊客中心" not in set(substitution["official_name"])
    assert set(substitution["substitution_status"]) == {"needs_manual_review"}
    assert "admin" not in set(substitution["site_id"])
