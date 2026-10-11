from __future__ import annotations

CATEGORY_REASONS = {
    "parser_error",
    "source_level_change",
    "official_reclassification_documented",
    "unresolved",
    "unchanged",
    "not_found",
}


def classify_category_discrepancy(
    csv_category: str,
    excel_category: str,
    excel_raw_cell: str,
    filled_by_parser: bool,
) -> str:
    """An implausible category name is not corrected. Blank cells are not filled from the previous block."""
    csv_label = (csv_category or "").strip()
    excel_label = (excel_category or "").strip()
    raw_cell = (excel_raw_cell or "").strip()
    if filled_by_parser and raw_cell == "" and excel_label and excel_label != csv_label:
        return "parser_error"
    if not csv_label and not excel_label:
        return "not_found"
    if csv_label and excel_label and csv_label == excel_label:
        return "unchanged"
    if csv_label and excel_label and csv_label != excel_label:
        return "unresolved"
    if csv_label and not excel_label:
        return "unresolved"
    return "unresolved"


def link_decision(
    found_2024: bool,
    same_name: bool,
    same_county: bool,
    same_category: bool,
    unique_candidate: bool,
    months_match_excel: bool,
) -> dict[str, object]:
    """Same name alone does not link a 2024 row, and a missing row is not imputed."""
    if not found_2024:
        return {
            "identity_status": "not_found",
            "safe_to_link": False,
            "continuity_evidence": "2024 沒有同縣市同名的原始列，不補值。",
            "category_change_reason": "not_found",
        }
    if not unique_candidate or not same_name or not same_county:
        return {
            "identity_status": "unresolved",
            "safe_to_link": False,
            "continuity_evidence": "同名候選不只一筆，或縣市、名稱沒有同時對上。",
            "category_change_reason": "unresolved",
        }
    if same_category and months_match_excel:
        return {
            "identity_status": "official_source_confirmed",
            "safe_to_link": True,
            "continuity_evidence": "2023 與 2024 的 CSV 類型相同，而且 12 個月人次與 Excel 同名列一致。",
            "category_change_reason": "unchanged",
        }
    if same_category:
        return {
            "identity_status": "rule_inferred",
            "safe_to_link": True,
            "continuity_evidence": "同縣市、同名、CSV 類型未變，年份接續。這是規則推定，不是人工確認。",
            "category_change_reason": "unchanged",
        }
    return {
        "identity_status": "rule_inferred_not_confirmed",
        "safe_to_link": False,
        "continuity_evidence": "同縣市同名，但類型來源不一致。同名不足以接續。",
        "category_change_reason": "unresolved",
    }


def alignment_pattern(csv_value: float, excel_value: float, previous_value: float | None, next_value: float | None) -> str:
    """Equal numbers describe a pattern. They do not by themselves prove the correct pairing."""
    if _same(csv_value, excel_value):
        return "same_row"
    if previous_value is not None and _same(csv_value, previous_value):
        return "equals_previous_visit_row"
    if next_value is not None and _same(csv_value, next_value):
        return "equals_next_visit_row"
    return "different"


def confirmed_parser_error(worksheet_value: float | None, parsed_value: float | None) -> bool:
    """A parser error requires the derived value to differ from the named row's own cell."""
    if worksheet_value is None or parsed_value is None:
        return False
    return not _same(float(worksheet_value), float(parsed_value))


def strict_asof_eligible(published_at: str) -> bool:
    text = (published_at or "").strip()
    return len(text) >= 10 and text[4] == "-" and text[7] == "-"


def valid_share_vector(shares: list[float], tolerance: float = 1e-8) -> bool:
    if not shares:
        return False
    if any(value < -tolerance for value in shares):
        return False
    return abs(sum(shares) - 1.0) <= tolerance


def exclude_holdout_rows(rows: list[dict[str, object]], holdout_year: int = 2025) -> tuple[list[dict[str, object]], list[str]]:
    kept = []
    held = []
    for row in rows:
        year = int(str(row["year_month"])[:4])
        if year >= holdout_year:
            held.append(str(row["year_month"]))
        else:
            kept.append(row)
    return kept, held


def _same(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-6
