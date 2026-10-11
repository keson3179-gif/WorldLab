from __future__ import annotations

import pandas as pd
import pytest

from worldlab_data.phase9 import (
    classify_2021_site_year,
    continuity_rows,
    measurement_applicability,
    observed_months,
    phase9_match_status,
)


def test_2021_published_total_is_january_to_october_and_is_not_called_a_revision() -> None:
    months = [float(value) for value in range(1, 13)]
    published = float(sum(months[:10]))
    diagnosis = classify_2021_site_year(months, published, months, float(sum(months)), "present")
    assert diagnosis["root_cause"] == "csv_published_equals_jan_to_oct_months_match_excel"
    assert diagnosis["unresolved"] is False
    assert diagnosis["official_revision_claimed"] is False
    assert months == [float(value) for value in range(1, 13)]


def test_november_and_december_differences_stay_unresolved() -> None:
    months = [10.0] * 12
    excel = [10.0] * 10 + [4.0, 6.0]
    published = 100.0
    diagnosis = classify_2021_site_year(months, published, excel, float(sum(excel)), "present")
    assert diagnosis["root_cause"] == "csv_published_equals_jan_to_oct_november_december_differ"
    assert diagnosis["unresolved"] is True
    assert diagnosis["official_revision_claimed"] is False
    assert "官方修訂" in str(diagnosis["evidence"])


def test_zero_months_do_not_hide_the_october_prefix() -> None:
    months = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]
    diagnosis = classify_2021_site_year(months, 5.0, None, None, "absent")
    assert 10 in diagnosis["matched_prefix_months"]
    assert diagnosis["root_cause"] == "csv_published_equals_jan_to_oct_excel_unmatched"


def test_parent_blank_continuation_keeps_the_original_id() -> None:
    rows = continuity_rows(
        [
            {
                "site_id": "stat_old",
                "official_name": "貓鼻頭公園",
                "county": "屏東縣",
                "parent_area": "墾丁國家公園",
                "category": "國家公園",
                "years": (2021,),
                "last_year": 2021,
            }
        ],
        [
            {
                "site_id": "stat_new",
                "official_name": "貓鼻頭公園",
                "county": "屏東縣",
                "parent_area": "-",
                "category": "國家公園",
                "years": (2022, 2023),
                "last_year": 2023,
            }
        ],
        years=(2022, 2023),
    )
    assert [row["old_site_id"] for row in rows] == ["stat_old", "stat_old"]
    assert {row["new_site_id_candidate"] for row in rows} == {"stat_new"}
    assert {row["continuity_status"] for row in rows} == {"versioned_same_name_parent_blank"}
    assert len({(row["old_site_id"], row["year"]) for row in rows}) == len(rows)


def test_category_change_and_same_year_duplicates_stay_unconfirmed() -> None:
    old = {
        "site_id": "stat_old",
        "official_name": "野柳地質公園",
        "county": "新北市",
        "parent_area": "北海岸及觀音山國家風景區",
        "category": "國家級風景特定區",
        "years": (2021,),
        "last_year": 2021,
    }
    changed = continuity_rows(
        [old],
        [
            {
                "site_id": "stat_2024",
                "official_name": "野柳地質公園",
                "county": "新北市",
                "parent_area": "-",
                "category": "其他",
                "years": (2024,),
                "last_year": 2024,
            }
        ],
        years=(2024,),
    )
    assert changed[0]["continuity_status"] == "category_changed_needs_manual_review"
    assert changed[0]["old_site_id"] == "stat_old"
    duplicate = continuity_rows(
        [old],
        [
            {**old, "site_id": "stat_a", "parent_area": "-", "years": (2022,), "last_year": 2022},
            {**old, "site_id": "stat_b", "parent_area": "-", "years": (2022,), "last_year": 2022},
        ],
        years=(2022,),
    )
    assert len(duplicate) == 1
    assert duplicate[0]["continuity_status"] == "ambiguous_same_year_needs_manual_review"


def test_holdout_year_does_not_change_the_common_window() -> None:
    base = [
        {"site_id": "a", "year": 2024, "year_month": "2024-12", "observation_status": "reported"},
        {"site_id": "b", "year": 2024, "year_month": "2024-12", "observation_status": "reported"},
    ]
    with_holdout = base + [
        {"site_id": "a", "year": 2025, "year_month": "2025-01", "observation_status": "reported"},
        {"site_id": "b", "year": 2025, "year_month": "2025-06", "observation_status": "reported"},
    ]
    without, excluded_without = observed_months(base, {"a"})
    with_rows, excluded_with = observed_months(with_holdout, {"a"})
    assert without == with_rows == {"2024-12"}
    assert excluded_without == 0
    assert excluded_with == 1
    left, _ = observed_months(with_holdout, {"a"})
    right, _ = observed_months(with_holdout, {"b"})
    assert left & right == {"2024-12"}


def test_measurement_label_is_traceable_but_not_site_evidence() -> None:
    labeled = measurement_applicability("電子計數器")
    blank = measurement_applicability("")
    telecom = measurement_applicability("電信大數據")
    assert labeled["applicability"] == "label_only_unchecked"
    assert labeled["document_evidence_for_this_site_year"] is False
    assert labeled["vocabulary_relation"] == "exact_phrase_in_2026_guideline"
    assert blank["applicability"] == "no_measurement_information"
    assert telecom["vocabulary_relation"] == "label_not_exact_guideline_phrase"


def test_phase9_does_not_promote_an_exact_rule_to_human_confirmation() -> None:
    assert phase9_match_status("confirmed_manual", "exact_unique_same_county", False) == "confirmed_exact_rule"
    assert phase9_match_status("needs_manual_review", "fuzzy_candidate", False) == "needs_manual_review"
    assert phase9_match_status("needs_manual_review", "no_candidate_above_floor", False) == "unmatched"
    assert phase9_match_status("confirmed_manual", "exact_unique_same_county", True) == "confirmed_manual"


def test_workspace_csv_mismatches_are_only_the_2021_october_prefix() -> None:
    from worldlab_data.config import load_paths
    from worldlab_data.phase9 import matching_prefixes
    from worldlab_data.phase9_run import _csv_year_rows, _wide

    paths = load_paths()
    if not paths.raw_root.exists():
        pytest.skip("政府公開資料不在這個工作目錄")
    try:
        wide, _path = _wide(paths)
    except StopIteration:
        pytest.skip("月別 CSV 不在這個工作目錄")
    mismatched = []
    for row in _csv_year_rows(wide):
        if row["months"] is None or row["published_annual"] is None:
            continue
        if abs(sum(row["months"]) - float(row["published_annual"])) <= 1e-6:
            continue
        mismatched.append(row)
    assert {row["year"] for row in mismatched} == {2021}
    assert len(mismatched) == 317
    assert all(sum(row["months"]) > float(row["published_annual"]) for row in mismatched)
    assert all(10 in matching_prefixes(row["months"], float(row["published_annual"])) for row in mismatched)
    assert isinstance(wide, pd.DataFrame)
