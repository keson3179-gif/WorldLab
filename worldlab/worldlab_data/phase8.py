from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from worldlab_data.textutil import comparison_key, county_parts, nfkc, parse_calendar_year, parse_count

MISSING_PARENTS = {"", "-", "—", "–", "無", "na", "n/a", "none"}
ADMIN_TOKENS = ("遊客中心", "遊客服務", "處本部", "管理處", "展示館", "服務中心")
AQUARIUM_TOKENS = ("海洋世界", "水族館")
OFFSHORE = {"綠島", "蘭嶼", "小琉球"}


def chinese_line(value: object) -> str:
    text = nfkc(value)
    if not text:
        return ""
    return text.split("\n")[0].strip()


def function_tag(name: object) -> str:
    text = nfkc(name)
    if any(token in text for token in ADMIN_TOKENS):
        return "admin_or_facility"
    if any(token in text for token in AQUARIUM_TOKENS):
        return "aquarium"
    if any(token in text for token in OFFSHORE):
        return "offshore_island"
    return "scenic_point_unverified"


def confirmation_decision(candidates: list[dict[str, object]]) -> tuple[str, str, str]:
    """Confirm only a single same-county exact name with coordinates. Fuzzy scores stay unconfirmed."""
    usable = [row for row in candidates if row.get("match_method") != "no_candidate_above_floor" and row.get("candidate_poi_id")]
    methods = {str(row.get("match_method")) for row in usable}
    if len(usable) == 1 and methods == {"exact_unique_same_county"} and usable[0].get("same_key") and usable[0].get("candidate_latitude") and usable[0].get("candidate_longitude"):
        note = "同縣市只有一筆景點，正規化名稱相同，且有座標。"
        if nfkc(usable[0].get("official_name")) != nfkc(usable[0].get("candidate_name")):
            note += "表面字形不同，但是正規化後相同，例如臺與台。"
        return "confirmed_manual", "無第二個候選", note
    if not usable:
        return "needs_manual_review", "沒有達到門檻的同縣市候選", "不要用較低分或其他縣市的景點補上。"
    if "exact_name_multiple_pois" in methods:
        return "needs_manual_review", "同名景點有多筆，座標不完全相同", "不能挑選其中一筆。"
    if methods == {"fuzzy_candidate"}:
        return "needs_manual_review", "只有模糊候選，名稱並非唯一同名", "分數再高也不視為已確認。"
    return "needs_manual_review", "候選條件不足以確認是同一據點", "保留待審。"


def classify_annual_difference(
    month_statuses: list[str],
    month_sum: float | None,
    published_annual: float | None,
    excel_state: str,
    excel_month_sum: float | None,
    excel_annual: float | None,
) -> str:
    """Classify a within-file annual mismatch. Do not rebuild months from the annual total."""
    if any(status not in {"reported", "reported_zero"} for status in month_statuses):
        return "missing_month"
    if published_annual is None or month_sum is None:
        return "format_parse"
    if excel_state == "present":
        months_differ = excel_month_sum is not None and abs(excel_month_sum - month_sum) > 1e-6
        annual_differ = excel_annual is not None and abs(excel_annual - published_annual) > 1e-6
        if months_differ or annual_differ:
            return "version_difference"
    return "unknown"


def parse_annual_matrix(rows: list[list[object]], source_file: str) -> list[dict[str, object]]:
    """Read a bilingual annual sheet. Parent header rows are not visit counts."""
    header_index = None
    month_columns: dict[int, int] = {}
    total_index = None
    method_index = None
    year = None
    for index, row in enumerate(rows):
        found_months: list[tuple[int, int, int]] = []
        found_totals: list[tuple[int, int | None]] = []
        method_column = None
        for column, cell in enumerate(row):
            compact = nfkc(cell).replace("\n", "")
            if "計算方式" in compact:
                method_column = column
            total_matched = re.search(r"(\d+)年1月\s*[-–—到至]\s*12月", compact)
            if total_matched:
                found_totals.append((column, parse_calendar_year(total_matched.group(1))))
                continue
            matched = re.search(r"(\d+)年(\d+)月", compact)
            if matched is None:
                continue
            month = int(matched.group(2))
            parsed_year = parse_calendar_year(matched.group(1))
            if parsed_year is None or not 1 <= month <= 12:
                continue
            found_months.append((column, parsed_year, month))
        primary_hits = [item for item in found_months if item[1] == found_months[0][1]] if found_months else []
        if len(primary_hits) >= 6:
            header_index = index
            kept: dict[int, int] = {}
            for column, _, month in primary_hits:
                if month not in kept.values():
                    kept[column] = month
            month_columns = kept
            year = primary_hits[0][1]
            total_index = next((column for column, total_year in found_totals if total_year == year), None)
            method_index = method_column
            break
    if header_index is None or year is None:
        return []
    parent = ""
    parsed: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        category = chinese_line(row[0] if len(row) > 0 else "")
        name = chinese_line(row[1] if len(row) > 1 else "")
        county, _ = county_parts(chinese_line(row[2] if len(row) > 2 else ""))
        month_cells = []
        for column, month in sorted(month_columns.items(), key=lambda item: item[1]):
            raw = row[column] if column < len(row) else None
            visits, status = parse_count(raw)
            month_cells.append((month, visits, status))
        numeric = [status in {"reported", "reported_zero"} for _, _, status in month_cells]
        method = chinese_line(row[method_index]) if method_index is not None and method_index < len(row) else ""
        if not county and name and not any(numeric):
            parent = name
            continue
        if not county or not name or not any(numeric):
            continue
        annual_value = None
        if total_index is not None and total_index < len(row):
            annual_value, _ = parse_count(row[total_index])
        for month, visits, status in month_cells:
            parsed.append(
                {
                    "source_file": source_file,
                    "year": year,
                    "month": month,
                    "year_month": f"{year:04d}-{month:02d}",
                    "visits": visits,
                    "observation_status": status,
                    "official_name": name,
                    "county": county,
                    "parent_area": parent,
                    "category": category,
                    "published_annual": annual_value,
                    "measurement_method": method,
                    "name_key": comparison_key(name),
                    "county_key": comparison_key(county),
                }
            )
    return parsed


def geographic_groups(history: pd.DataFrame) -> pd.DataFrame:
    """Same county and named scenic area. Size is not limited, and co-location is not substitution."""
    rows: list[dict[str, object]] = []
    if history.empty:
        return pd.DataFrame()
    for (county, parent), group in history.groupby(["county", "parent_area"]):
        if str(parent).strip().casefold() in MISSING_PARENTS:
            continue
        members = group.drop_duplicates("site_id")
        if len(members) < 2:
            continue
        group_id = f"geo_{_slug(str(county))}_{_slug(str(parent))}"
        for member in members.to_dict(orient="records"):
            rows.append(
                {
                    "definition": "A_geographic",
                    "group_id": group_id,
                    "county": county,
                    "parent_area": parent,
                    "site_id": member["site_id"],
                    "official_name": member["official_name"],
                    "function_tag": function_tag(member["official_name"]),
                    "substitution_status": "not_a_substitution_group",
                    "group_size": int(len(members)),
                }
            )
    return pd.DataFrame.from_records(rows)


def substitution_candidates(geographic: pd.DataFrame) -> pd.DataFrame:
    """Scenic-point clusters inside a geographic group. National-park membership is not confirmation."""
    if geographic.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    scenic = geographic[geographic["function_tag"] == "scenic_point_unverified"]
    for group_id, group in scenic.groupby("group_id"):
        if len(group) < 2:
            continue
        for member in group.to_dict(orient="records"):
            rows.append(
                {
                    **member,
                    "definition": "B_substitution_candidate",
                    "group_id": f"sub_{group_id}",
                    "substitution_status": "needs_manual_review",
                    "group_size": int(len(group)),
                    "exclusion_note": "同屬一個國家公園或風景區只表示地理分布，不表示活動可以互相替代。",
                }
            )
    return pd.DataFrame.from_records(rows)


def _slug(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
    text = "".join(character for character in value if character.isalnum())[:18]
    return f"{text}{digest}"
