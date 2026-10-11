from __future__ import annotations

from worldlab_data.phase10 import (
    assign_site_uids,
    classify_2021_months,
    comparability_level,
    derived_annual,
    parse_annual_sheet,
    usable_months,
)


def _sheet():
    header = ["類型", "觀光遊憩區", "縣市", "100年\n1月", "100年\n2月", "100年\n3月", "100年\n4月", "100年\n5月", "100年\n6月", "遊客人次計算方式"]
    parent = ["國家級風景特定區", "茂林國家風景區", "", "", "", "", "", "", "", ""]
    child = ["", "茂林遊憩區", "高雄市", 1, 1, 1, 1, 1, 1, "電子計數器"]
    spilled = ["公立", "七星潭風景區", "花蓮縣", 2, 2, 2, 2, 2, 2, "資通訊設備"]
    return [header, parent, child, spilled]


def test_parent_area_spillover() -> None:
    parsed = parse_annual_sheet(_sheet(), [], "sample.xlsx", "累計明細表")
    by_name = {row["official_name"]: row for row in parsed}
    assert by_name["茂林遊憩區"]["parent_area"] == "茂林國家風景區"
    assert by_name["茂林遊憩區"]["county"] == "高雄市"
    assert by_name["七星潭風景區"]["parent_area"] == "unknown"
    assert by_name["七星潭風景區"]["raw_parent_area"] == ""
    assert by_name["七星潭風景區"]["county"] == "花蓮縣"
    assert by_name["七星潭風景區"]["measurement_method"] == "資通訊設備"
    cultural = parse_annual_sheet(
        [
            ["類型", "觀光遊憩區", "縣市", "100年\n1月", "100年\n2月", "100年\n3月", "100年\n4月", "100年\n5月", "100年\n6月", "遊客人次計算方式"],
            ["國家級風景特定區", "茂林國家風景區", "", "", "", "", "", "", "", ""],
            ["", "茂林遊憩區", "高雄市", 1, 1, 1, 1, 1, 1, "電子計數器"],
            ["", "賽嘉遊憩區", "屏東縣", 1, 1, 1, 1, 1, 1, "電子計數器"],
            ["文教場所", "茂林國家風景區", "", "", "", "", "", "", "", ""],
            ["", "臺灣原住民文化園區", "屏東縣", 3, 3, 3, 3, 3, 3, "門票數"],
            ["", "臺灣客家文化館", "苗栗縣", 4, 4, 4, 4, 4, 4, "電信大數據"],
            ["", "六堆客家文化園區", "屏東縣", 5, 5, 5, 5, 5, 5, "門票數"],
        ],
        [],
        "sample.xlsx",
        "累計明細表",
    )
    cultural_by_name = {row["official_name"]: row for row in cultural}
    assert cultural_by_name["臺灣原住民文化園區"]["parent_area"] == "茂林國家風景區"
    assert cultural_by_name["臺灣客家文化館"]["parent_area"] == "unknown"
    assert cultural_by_name["臺灣客家文化館"]["raw_parent_area"] == "茂林國家風景區"
    assert cultural_by_name["臺灣客家文化館"]["measurement_method"] == "電信大數據"
    assert cultural_by_name["六堆客家文化園區"]["parent_area"] == "unknown"


def test_source_row_lineage() -> None:
    parsed = parse_annual_sheet(_sheet(), [], "sample.xlsx", "累計明細表")
    row = next(item for item in parsed if item["official_name"] == "七星潭風景區")
    assert row["source_sheet"] == "累計明細表"
    assert row["source_row_number"] == 4
    assert row["source_file"] == "sample.xlsx"
    assert "raw_parent_area" in row


def test_canonical_site_uid() -> None:
    rows = assign_site_uids(
        [
            {"raw_site_id": "old", "official_name": "貓鼻頭公園", "county": "屏東縣", "category": "國家公園", "parent_area": "墾丁國家公園", "first_year": 2012, "last_year": 2021, "months": {"2021-12": 3}},
            {"raw_site_id": "new", "official_name": "貓鼻頭公園", "county": "屏東縣", "category": "國家公園", "parent_area": "-", "first_year": 2022, "last_year": 2024, "months": {"2022-01": 4}},
            {"raw_site_id": "fuzzy", "official_name": "貓鼻頭", "county": "屏東縣", "category": "國家公園", "parent_area": "-", "first_year": 2022, "last_year": 2022, "months": {"2022-01": 9}},
            {"raw_site_id": "typed", "official_name": "貓鼻頭公園", "county": "屏東縣", "category": "其他", "parent_area": "-", "first_year": 2024, "last_year": 2024, "months": {"2024-01": 1}},
        ]
    )
    by_id = {row["raw_site_id"]: row for row in rows}
    assert by_id["old"]["site_uid"] == by_id["new"]["site_uid"]
    assert by_id["old"]["site_version_id"] != by_id["new"]["site_version_id"]
    assert by_id["old"]["match_status"] == "provisional_rule_match"
    assert by_id["fuzzy"]["site_uid"] != by_id["old"]["site_uid"]
    assert by_id["typed"]["site_uid"] != by_id["old"]["site_uid"]
    assert by_id["typed"]["review_note"]


def test_site_month_uniqueness() -> None:
    rows = assign_site_uids(
        [
            {"raw_site_id": "a", "official_name": "白沙灣", "county": "新北市", "category": "國家級風景特定區", "parent_area": "北海岸", "first_year": 2021, "last_year": 2021, "months": {"2022-01": 10}},
            {"raw_site_id": "b", "official_name": "白沙灣", "county": "新北市", "category": "國家級風景特定區", "parent_area": "-", "first_year": 2022, "last_year": 2022, "months": {"2022-01": 12}},
        ]
    )
    assert rows[0]["site_uid"] != rows[1]["site_uid"]


def test_2021_annual_derivation() -> None:
    months = [float(value) for value in range(1, 13)]
    published = float(sum(months[:10]))
    result = derived_annual(months, published)
    assert result["derived_annual_sum"] == float(sum(months))
    assert result["published_annual"] == published
    assert result["published_equals_january_to_october"] is True
    assert months == [float(value) for value in range(1, 13)]


def test_conflicting_month_versions() -> None:
    csv_months = [10.0] * 10 + [8.0, 9.0]
    excel_months = [10.0] * 10 + [3.0, 4.0]
    result = classify_2021_months(csv_months, excel_months)
    assert result["source_class"] == "B_november_december_conflict"
    assert result["source_conflict"] is True
    assert result["chosen_version"] == ""
    assert result["csv_november"] == 8.0
    assert result["excel_november"] == 3.0
    assert result["months_kept"] is True


def test_measurement_comparability_levels() -> None:
    direct = comparability_level([{"measurement_method": "電子計數器", "document_evidence_for_this_site_year": False}])
    proxy = comparability_level([{"measurement_method": "電信大數據", "document_evidence_for_this_site_year": False}])
    documented = comparability_level(
        [
            {"measurement_method": "電子計數器", "document_evidence_for_this_site_year": True},
            {"measurement_method": "電子計數器", "document_evidence_for_this_site_year": True},
        ]
    )
    assert direct["label"] == "reported_visits_conditional"
    assert direct["level"] == "B"
    assert proxy["label"] == "not_comparable"
    assert proxy["level"] == "C"
    assert documented["label"] == "documented_comparable"
    assert documented["level"] == "A"


def test_no_holdout_selection() -> None:
    base = {"2024-12"}
    with_holdout = {"2024-12", "2025-01", "2026-03"}
    assert usable_months(base, set()) == usable_months(with_holdout, set()) == {"2024-12"}
    assert "2025-01" not in usable_months(with_holdout, {"2024-12"})
