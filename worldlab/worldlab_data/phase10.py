from __future__ import annotations

import re
from collections import defaultdict

from worldlab_data.textutil import comparison_key, county_parts, nfkc, parse_calendar_year, parse_count, stable_site_id

UNKNOWN_PARENT = "unknown"
HOLDOUT_YEAR = 2025
PROXY_TOKENS = ("停車", "住宿", "電信", "概估", "推估", "用餐", "海、空運", "入境人次")
DIRECT_TOKENS = ("門票", "電子計數器", "柵欄", "AI精密人流", "人工計數", "計數器")


def chinese_line(value: object) -> str:
    text = nfkc(value)
    if not text:
        return ""
    return text.split("\n")[0].strip()


def category_by_row(rows: list[list[object]], merges: list[tuple[int, int, int, int]]) -> dict[int, str]:
    """Fill category only inside a column-A merged range. This is not a parent-area fill."""
    found: dict[int, str] = {}
    for min_row, max_row, min_col, max_col in merges:
        if min_col != 1 or max_col != 1 or min_row < 1:
            continue
        if min_row - 1 >= len(rows):
            continue
        label = chinese_line(rows[min_row - 1][0] if rows[min_row - 1] else "")
        if not label:
            continue
        for excel_row in range(min_row, max_row + 1):
            found[excel_row] = label
    return found


def _header_columns(rows: list[list[object]]) -> tuple[int | None, dict[int, int], int | None, int | None, int | None]:
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
        primary = [item for item in found_months if item[1] == found_months[0][1]] if found_months else []
        if len(primary) < 6:
            continue
        kept: dict[int, int] = {}
        for column, _, month in primary:
            if month not in kept.values():
                kept[column] = month
        year = primary[0][1]
        total_index = next((column for column, total_year in found_totals if total_year == year), None)
        return index, kept, total_index, method_column, year
    return None, {}, None, None, None


def parse_annual_sheet(
    rows: list[list[object]],
    merges: list[tuple[int, int, int, int]],
    source_file: str,
    source_sheet: str,
) -> list[dict[str, object]]:
    """Parse one annual sheet. A parent header never crosses into the next category block."""
    header_index, month_columns, total_index, method_index, year = _header_columns(rows)
    if header_index is None or year is None:
        return []
    merged_category = category_by_row(rows, merges)
    carried_category = ""
    parent = ""
    parent_category = ""
    parent_row = None
    parsed: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        excel_row = index + 1
        if index <= header_index:
            continue
        explicit_category = chinese_line(row[0] if row else "")
        if explicit_category:
            carried_category = explicit_category
            category = explicit_category
        elif excel_row in merged_category:
            category = merged_category[excel_row]
        elif merges:
            category = ""
        else:
            category = carried_category
        name = chinese_line(row[1] if len(row) > 1 else "")
        county, _ = county_parts(chinese_line(row[2] if len(row) > 2 else ""))
        month_cells = []
        for column, month in sorted(month_columns.items(), key=lambda item: item[1]):
            raw = row[column] if column < len(row) else None
            visits, status = parse_count(raw)
            month_cells.append((month, visits, status))
        numeric = [status in {"reported", "reported_zero"} for _, _, status in month_cells]
        method = chinese_line(row[method_index]) if method_index is not None and method_index < len(row) else ""
        if name and not county and not any(numeric):
            parent = name
            parent_category = category
            parent_row = excel_row
            continue
        if not county or not name or not any(numeric):
            continue
        same_block = bool(parent) and bool(category) and category == parent_category
        if same_block:
            resolved_parent = parent
            raw_parent = parent
            resolution = "same_category_header"
            header_row = parent_row
        else:
            resolved_parent = UNKNOWN_PARENT
            raw_parent = ""
            resolution = "unknown_no_header_in_category"
            header_row = None
        annual_value = None
        if total_index is not None and total_index < len(row):
            annual_value, _ = parse_count(row[total_index])
        parsed.append(
            {
                "source_file": source_file,
                "source_sheet": source_sheet,
                "source_row_number": excel_row,
                "year": year,
                "official_name": name,
                "county": county,
                "category": category,
                "raw_category_cell": explicit_category,
                "measurement_method": method,
                "raw_parent_area": raw_parent,
                "parent_area": resolved_parent,
                "parent_header_row": header_row or "",
                "parent_resolution": resolution,
                "published_annual": annual_value,
                "months": {month: visits for month, visits, status in month_cells if status in {"reported", "reported_zero"} and visits is not None},
                "name_key": comparison_key(name),
                "county_key": comparison_key(county),
            }
        )
    _limit_parent_to_home_counties(parsed)
    return parsed


def _limit_parent_to_home_counties(parsed: list[dict[str, object]]) -> None:
    """A scenic-area header does not cover later rows in another county.

    Home counties are the counties already observed under that header inside a scenic or
    national-park category. The first child outside that set, and every later row under
    the same header, becomes unknown. The raw header text stays in raw_parent_area.
    """
    home: dict[str, set[str]] = defaultdict(set)
    for row in parsed:
        parent = str(row["parent_area"])
        category = str(row["category"])
        if parent == UNKNOWN_PARENT:
            continue
        if "風景" in category or "國家公園" in category:
            home[parent].add(str(row["county"]))
    blocked: set[tuple[str, object]] = set()
    for row in parsed:
        parent = str(row["parent_area"])
        counties = home.get(parent)
        if not counties:
            continue
        header_key = (parent, row.get("parent_header_row"))
        if header_key in blocked or str(row["county"]) not in counties:
            blocked.add(header_key)
            row["parent_area"] = UNKNOWN_PARENT
            row["parent_resolution"] = "unknown_outside_header_counties"


def assign_site_uids(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group raw ids only when county, normalized name, and category agree and the years abut."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            comparison_key(record.get("county")),
            comparison_key(record.get("official_name")),
            comparison_key(record.get("category")),
        )
        grouped[key].append(record)
    rows: list[dict[str, object]] = []
    for key, members in grouped.items():
        ordered = sorted(members, key=lambda item: (int(item.get("first_year") or 0), str(item.get("raw_site_id"))))
        chain: list[dict[str, object]] = []
        for member in ordered:
            if not chain:
                chain = [member]
                continue
            if _can_extend(chain, member):
                chain.append(member)
                continue
            rows.extend(_emit_chain(chain))
            chain = [member]
        if chain:
            rows.extend(_emit_chain(chain))
    categories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        categories[(comparison_key(record.get("county")), comparison_key(record.get("official_name")))].add(comparison_key(record.get("category")))
    for row in rows:
        key = (comparison_key(row.get("county")), comparison_key(row.get("official_name")))
        if len(categories[key]) > 1:
            row["match_status"] = "needs_manual_review" if row["match_status"] == "singleton" else row["match_status"]
            row["review_note"] = "同縣市同名另有不同類型。那些類型沒有合成這個 site_uid，也沒有用模糊名稱確定身分。"
        else:
            row["review_note"] = ""
    return rows


def _can_extend(chain: list[dict[str, object]], member: dict[str, object]) -> bool:
    previous = chain[-1]
    if int(member.get("first_year") or 0) != int(previous.get("last_year") or 0) + 1:
        return False
    if _months_conflict(chain, member):
        return False
    return True


def _months_conflict(chain: list[dict[str, object]], member: dict[str, object]) -> bool:
    owned: dict[str, float] = {}
    for item in chain:
        for year_month, visits in dict(item.get("months") or {}).items():
            owned[str(year_month)] = float(visits)
    for year_month, visits in dict(member.get("months") or {}).items():
        if str(year_month) in owned and abs(owned[str(year_month)] - float(visits)) > 1e-6:
            return True
    return False


def _emit_chain(chain: list[dict[str, object]]) -> list[dict[str, object]]:
    anchor = chain[0]
    site_uid = stable_site_id(
        [
            "uid",
            str(anchor.get("county")),
            str(anchor.get("official_name")),
            str(anchor.get("category")),
            str(anchor.get("raw_site_id")),
        ]
    )
    status = "provisional_rule_match" if len(chain) > 1 else "singleton"
    evidence = (
        "同縣市、正規化名稱相同、類型相同，而且年份接續。沒有用模糊名稱合併。"
        if len(chain) > 1
        else "這個原始 ID 沒有和另一個 ID 合併。"
    )
    emitted = []
    for member in chain:
        emitted.append(
            {
                "raw_site_id": member.get("raw_site_id"),
                "site_uid": site_uid,
                "site_version_id": member.get("raw_site_id"),
                "official_name": member.get("official_name"),
                "county": member.get("county"),
                "category": member.get("category"),
                "parent_area": member.get("parent_area"),
                "match_status": status,
                "evidence": evidence,
                "first_year": member.get("first_year"),
                "last_year": member.get("last_year"),
            }
        )
    return emitted


def _review_category_changes(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Same name and county with a different category stay unmerged. No fuzzy confirmation."""
    by_place: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        by_place[(comparison_key(record.get("county")), comparison_key(record.get("official_name")))].add(
            comparison_key(record.get("category"))
        )
    notes = []
    seen = set()
    for record in records:
        key = (comparison_key(record.get("county")), comparison_key(record.get("official_name")))
        if len(by_place[key]) < 2 or key in seen:
            continue
        seen.add(key)
        notes.append(
            {
                "raw_site_id": "",
                "site_uid": "",
                "site_version_id": "",
                "official_name": record.get("official_name"),
                "county": record.get("county"),
                "category": "",
                "parent_area": "",
                "match_status": "needs_manual_review",
                "evidence": "同縣市同名出現一種以上的類型。沒有合成同一個 site_uid，也沒有用模糊名稱確定身分。",
                "first_year": "",
                "last_year": "",
            }
        )
    return notes


def derived_annual(months: list[float], published_annual: float) -> dict[str, object]:
    """Keep the published total and add a twelve-month sum. Do not rewrite month values."""
    original = list(months)
    derived = float(sum(months))
    if months != original:
        raise RuntimeError("月份原值被改寫")
    return {
        "published_annual": published_annual,
        "derived_annual_sum": derived,
        "published_equals_january_to_october": abs(float(sum(months[:10])) - float(published_annual)) <= 1e-6,
    }


def classify_2021_months(
    csv_months: list[float],
    excel_months: list[float] | None,
) -> dict[str, object]:
    """Keep both November and December values when the sources disagree. Do not drop the year."""
    if excel_months is None:
        return {
            "source_class": "C_excel_unmatched",
            "source_conflict": False,
            "chosen_version": "",
            "months_kept": True,
        }
    first_10 = all(abs(float(left) - float(right)) <= 1e-6 for left, right in zip(csv_months[:10], excel_months[:10]))
    nov_dec = abs(float(csv_months[10]) - float(excel_months[10])) <= 1e-6 and abs(float(csv_months[11]) - float(excel_months[11])) <= 1e-6
    if first_10 and nov_dec:
        source_class = "A_months_agree"
        conflict = False
    elif first_10:
        source_class = "B_november_december_conflict"
        conflict = True
    else:
        source_class = "B_other_month_conflict"
        conflict = True
    return {
        "source_class": source_class,
        "source_conflict": conflict,
        "chosen_version": "",
        "csv_november": csv_months[10],
        "excel_november": excel_months[10],
        "csv_december": csv_months[11],
        "excel_december": excel_months[11],
        "months_kept": True,
    }


def method_family(label: object) -> str:
    text = "" if label is None else str(label)
    if not text.strip():
        return "missing"
    if any(token in text for token in PROXY_TOKENS):
        return "proxy"
    if any(token in text for token in DIRECT_TOKENS):
        return "direct_counter"
    return "unclassified"


def comparability_level(member_years: list[dict[str, object]]) -> dict[str, str]:
    """Same method text is not enough for documented_comparable."""
    if not member_years:
        return {"level": "C", "label": "not_comparable", "reason": "沒有可判定的據點年。"}
    families = {method_family(item.get("measurement_method")) for item in member_years}
    documented = all(bool(item.get("document_evidence_for_this_site_year")) for item in member_years)
    changed = len({str(item.get("measurement_method") or "") for item in member_years if str(item.get("measurement_method") or "")}) > 1
    if "proxy" in families or "missing" in families and "direct_counter" not in families and "unclassified" not in families and not documented:
        return {
            "level": "C",
            "label": "not_comparable",
            "reason": "計數標籤是停車、住宿、電信或其他代理，或完全沒有標籤。只能描述個別序列。",
        }
    if "proxy" in families:
        return {
            "level": "C",
            "label": "not_comparable",
            "reason": "群內同時有到訪計數與代理估計，單位不能放在同一個份額裡。",
        }
    if documented and not changed and families <= {"direct_counter"}:
        return {
            "level": "A",
            "label": "documented_comparable",
            "reason": "每個據點年都有文件把方法指到該年，而且不是代理估計。",
        }
    if families & {"direct_counter", "unclassified", "missing"} and "proxy" not in families:
        limits = "方法名稱相同也不視為完全可比。"
        if changed:
            limits += "同一據點的標籤跨年不同。"
        if "missing" in families or "unclassified" in families:
            limits += "有年份沒有可歸類的直接計數標籤。"
        return {
            "level": "B",
            "label": "reported_visits_conditional",
            "reason": "官方欄位是回報人次，可以做探索性份額，但沒有據點級文件。" + limits,
        }
    return {"level": "C", "label": "not_comparable", "reason": "現有證據不足以把這些人次放在一起比較。"}


def usable_months(year_months: set[str], conflict_months: set[str], holdout_year: int = HOLDOUT_YEAR) -> set[str]:
    """Holdout months are excluded and cannot change which earlier months remain."""
    kept = set()
    for year_month in year_months:
        year_text = str(year_month)[:4]
        if year_text.isdigit() and int(year_text) >= holdout_year:
            continue
        if year_month in conflict_months:
            continue
        kept.add(year_month)
    return kept
