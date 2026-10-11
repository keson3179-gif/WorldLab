from __future__ import annotations

from collections import defaultdict

from rapidfuzz import fuzz

from worldlab_data.textutil import comparison_key

# Article 6 of the consolidated operations note, amended 2025-09-18 and effective 2026-01-01.
# The note names allowable method families. It does not assign a method to a site-year.
GUIDELINE_EFFECTIVE_DATE = "2026-01-01"
GUIDELINE_METHOD_PHRASES = (
    "門票數",
    "停車數",
    "住宿人次",
    "海、空運入境人次",
    "資通訊設備",
    "電子計數器",
    "柵欄式計數設備",
    "AI精密人流偵測",
    "電信數據",
)
HOLDOUT_YEAR = 2025
CONTINUITY_YEARS = (2022, 2023, 2024)


def prefix_sums(months: list[float]) -> list[float]:
    running = 0.0
    totals: list[float] = []
    for value in months:
        running += float(value)
        totals.append(running)
    return totals


def matching_prefixes(months: list[float], published: float) -> list[int]:
    return [index + 1 for index, total in enumerate(prefix_sums(months)) if abs(total - float(published)) <= 1e-6]


def classify_2021_site_year(
    months: list[float],
    published: float,
    excel_months: list[float] | None,
    excel_annual: float | None,
    excel_state: str,
) -> dict[str, object]:
    """Explain a 2021 total mismatch. Do not rewrite months or call a difference an official revision."""
    original = list(months)
    totals = prefix_sums(months)
    matched = matching_prefixes(months, published)
    month_sum = totals[-1] if totals else None
    prefix_10 = totals[9] if len(totals) >= 10 else None
    equals_jan_oct = prefix_10 is not None and abs(prefix_10 - float(published)) <= 1e-6
    november_delta = None
    december_delta = None
    first_10_equal = None
    if excel_months is not None and len(excel_months) == 12 and len(months) == 12:
        first_10_equal = all(abs(float(left) - float(right)) <= 1e-6 for left, right in zip(months[:10], excel_months[:10]))
        november_delta = float(months[10]) - float(excel_months[10])
        december_delta = float(months[11]) - float(excel_months[11])
    nov_dec_differ = november_delta is not None and (abs(november_delta) > 1e-6 or abs(december_delta or 0) > 1e-6)
    if not equals_jan_oct:
        root = "unresolved_not_january_to_october"
        unresolved = True
    elif excel_state == "present" and first_10_equal and not nov_dec_differ:
        root = "csv_published_equals_jan_to_oct_months_match_excel"
        unresolved = False
    elif excel_state == "present" and first_10_equal and nov_dec_differ:
        root = "csv_published_equals_jan_to_oct_november_december_differ"
        unresolved = True
    elif excel_state == "absent":
        root = "csv_published_equals_jan_to_oct_excel_unmatched"
        unresolved = True
    else:
        root = "csv_published_equals_jan_to_oct_unresolved_pair"
        unresolved = True
    evidence = _annual_evidence(matched, prefix_10, published, month_sum, excel_state, excel_annual, first_10_equal, november_delta, december_delta)
    if months != original:
        raise RuntimeError("月份原值被改寫")
    return {
        "root_cause": root,
        "evidence": evidence,
        "unresolved": unresolved,
        "official_revision_claimed": False,
        "matched_prefix_months": matched,
        "prefix_1_to_10": prefix_10,
        "month_sum": month_sum,
        "first_10_months_equal_excel": first_10_equal,
        "november_delta_csv_minus_excel": november_delta,
        "december_delta_csv_minus_excel": december_delta,
    }


def _annual_evidence(
    matched: list[int],
    prefix_10: float | None,
    published: float,
    month_sum: float | None,
    excel_state: str,
    excel_annual: float | None,
    first_10_equal: bool | None,
    november_delta: float | None,
    december_delta: float | None,
) -> str:
    parts = [f"CSV 公布合計 {published:g}，十二個月合計 {None if month_sum is None else format(month_sum, 'g')}。"]
    if 10 in matched and prefix_10 is not None:
        extra = "" if matched == [10] else f"同時等於前綴 {matched}，是因為這些前綴之間的月份為 0。"
        parts.append(f"公布合計等於 1–10 月累計 {prefix_10:g}。{extra}")
    else:
        parts.append(f"公布合計不等於 1–10 月累計。對上的前綴月份是 {matched or '無'}。")
    if excel_state == "present" and first_10_equal:
        parts.append("Excel 同年 1–10 月與 CSV 相同。")
        if november_delta is not None and abs(november_delta) <= 1e-6 and december_delta is not None and abs(december_delta) <= 1e-6:
            parts.append("11 月與 12 月也相同。Excel 年度合計是 Excel 自己的十二個月合計，不是 CSV 的公布合計。")
        else:
            parts.append(f"11 月差額 {november_delta:g}，12 月差額 {december_delta:g}。沒有修訂紀錄，不能把這個差額說成官方修訂。")
    elif excel_state == "absent":
        parts.append("年度 Excel 沒有同縣市同名的列。")
    elif excel_state == "present":
        parts.append("Excel 有同名列，但 1–10 月並非逐月相同。")
    if excel_annual is not None:
        parts.append(f"Excel 年度合計 {excel_annual:g}。")
    parts.append("月份原值沒有改寫。version_difference 只是差異分類，不是已查到的版本修訂。")
    return "".join(parts)


def phase9_match_status(phase8_status: str, match_method: str, human_reviewed: bool = False) -> str:
    """Phase 8 labels stay historical. This round does not invent a human confirmation."""
    if human_reviewed and phase8_status == "confirmed_manual":
        return "confirmed_manual"
    if phase8_status == "confirmed_manual" and match_method == "exact_unique_same_county":
        return "confirmed_exact_rule"
    if match_method == "no_candidate_above_floor" or phase8_status == "unmatched":
        return "unmatched"
    return "needs_manual_review"


def continuity_rows(old_sites: list[dict[str, object]], later_sites: list[dict[str, object]], years: tuple[int, ...] = CONTINUITY_YEARS) -> list[dict[str, object]]:
    """Link a later same-name record without replacing the original site_id or splicing visits."""
    by_exact: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_county: dict[str, list[dict[str, object]]] = defaultdict(list)
    for site in later_sites:
        by_exact[(comparison_key(site.get("county")), comparison_key(site.get("official_name")))].append(site)
        by_county[comparison_key(site.get("county"))].append(site)
    rows: list[dict[str, object]] = []
    for old in old_sites:
        old_last = int(old["last_year"])
        old_years = {int(year) for year in old.get("years") or []}
        confirmed_through: dict[str, int] = {}
        for year in years:
            if year in old_years:
                rows.append(
                    {
                        "old_site_id": old.get("site_id"),
                        "new_site_id_candidate": old.get("site_id"),
                        "old_name": old.get("official_name"),
                        "new_name": old.get("official_name"),
                        "county": old.get("county"),
                        "year": year,
                        "parent_area_before": old.get("parent_area"),
                        "parent_area_after": old.get("parent_area"),
                        "category_before": old.get("category"),
                        "category_after": old.get("category"),
                        "continuity_status": "same_site_id_continues",
                        "evidence": f"{year} 年仍使用原來的 site_id。",
                    }
                )
                continue
            candidates = [
                site
                for site in by_exact[(comparison_key(old.get("county")), comparison_key(old.get("official_name")))]
                if site.get("site_id") != old.get("site_id") and year in set(site.get("years") or [])
            ]
            row = _one_year_link(old, year, old_last, candidates, by_county[comparison_key(old.get("county"))], confirmed_through)
            if row["continuity_status"] == "versioned_same_name_parent_blank":
                confirmed_through[str(row["new_site_id_candidate"])] = year
            rows.append(row)
    _downgrade_merges(rows)
    return rows


def _one_year_link(
    old: dict[str, object],
    year: int,
    old_last: int,
    candidates: list[dict[str, object]],
    county_pool: list[dict[str, object]],
    confirmed_through: dict[str, int],
) -> dict[str, object]:
    base = {
        "old_site_id": old.get("site_id"),
        "new_site_id_candidate": "",
        "old_name": old.get("official_name"),
        "new_name": "",
        "county": old.get("county"),
        "year": year,
        "parent_area_before": old.get("parent_area"),
        "parent_area_after": "",
        "category_before": old.get("category"),
        "category_after": "",
        "continuity_status": "not_found_same_name",
        "evidence": f"{year} 年同縣市沒有相同名稱。原始 ID 保留，不拼接人次。",
    }
    if len(candidates) > 1:
        base.update(
            {
                "new_site_id_candidate": "|".join(str(site.get("site_id")) for site in candidates),
                "new_name": "|".join(str(site.get("official_name")) for site in candidates),
                "continuity_status": "ambiguous_same_year_needs_manual_review",
                "evidence": f"{year} 年同縣市同名有 {len(candidates)} 個 site_id，不能選其中一個。",
            }
        )
        return base
    if len(candidates) == 1:
        new = candidates[0]
        parent_changed = comparison_key(old.get("parent_area")) != comparison_key(new.get("parent_area"))
        category_same = comparison_key(old.get("category")) == comparison_key(new.get("category"))
        abuts = year == old_last + 1 or confirmed_through.get(str(new.get("site_id"))) == year - 1
        base.update(
            {
                "new_site_id_candidate": new.get("site_id"),
                "new_name": new.get("official_name"),
                "parent_area_after": new.get("parent_area"),
                "category_after": new.get("category"),
            }
        )
        if parent_changed and category_same and abuts:
            base["continuity_status"] = "versioned_same_name_parent_blank"
            base["evidence"] = (
                f"同縣市同名，{year} 年只有這個 site_id。"
                f"遊憩區由「{old.get('parent_area')}」變為「{new.get('parent_area')}」，類型仍是「{new.get('category')}」。"
                "site_id 含遊憩區，所以編號改變。原始 ID 保留，不拼接人次。"
            )
        elif not category_same:
            base["continuity_status"] = "category_changed_needs_manual_review"
            base["evidence"] = (
                f"同縣市同名，但類型由「{old.get('category')}」變為「{new.get('category')}」。"
                "沒有把它當成已確認的同一據點，也不拼接人次。"
            )
        else:
            base["continuity_status"] = "gapped_same_name_needs_manual_review"
            base["evidence"] = f"同縣市同名，但與上一個已觀察年份不接續（舊據點最後年到 {old_last}）。維持待審，不拼接。"
        return base
    fuzzy = _fuzzy_candidates(old, year, county_pool)
    if len(fuzzy) == 1:
        score, site = fuzzy[0]
        base.update(
            {
                "new_site_id_candidate": site.get("site_id"),
                "new_name": site.get("official_name"),
                "parent_area_after": site.get("parent_area"),
                "category_after": site.get("category"),
                "continuity_status": "rename_candidate_needs_manual_review",
                "evidence": f"同縣市沒有相同名稱。最接近的名稱是「{site.get('official_name')}」，分數 {score}。不視為已確認改名。",
            }
        )
    elif len(fuzzy) > 1:
        base["continuity_status"] = "ambiguous_same_year_needs_manual_review"
        base["new_name"] = "|".join(str(site.get("official_name")) for _, site in fuzzy[:3])
        base["evidence"] = f"{year} 年同縣市有多個接近名稱，不能改名拼接。"
    return base


def _fuzzy_candidates(old: dict[str, object], year: int, county_pool: list[dict[str, object]]) -> list[tuple[int, dict[str, object]]]:
    found = []
    old_key = comparison_key(old.get("official_name"))
    for site in county_pool:
        if site.get("site_id") == old.get("site_id") or year not in set(site.get("years") or []):
            continue
        if comparison_key(site.get("official_name")) == old_key:
            continue
        score = int(fuzz.ratio(old_key, comparison_key(site.get("official_name"))))
        if score >= 90:
            found.append((score, site))
    return sorted(found, key=lambda item: item[0], reverse=True)


def _downgrade_merges(rows: list[dict[str, object]]) -> None:
    owners: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        new_id = str(row.get("new_site_id_candidate") or "")
        if row.get("continuity_status") == "versioned_same_name_parent_blank" and new_id and "|" not in new_id:
            owners[(new_id, int(row["year"]))].add(str(row.get("old_site_id")))
    merged = {key for key, olds in owners.items() if len(olds) > 1}
    for row in rows:
        key = (str(row.get("new_site_id_candidate") or ""), int(row["year"]))
        if key in merged and row.get("continuity_status") == "versioned_same_name_parent_blank":
            row["continuity_status"] = "merge_needs_manual_review"
            row["evidence"] = "同一個新 site_id 對上了兩個以上的舊據點。可能是合併，維持待審，不拼接。"


def observed_months(visits: list[dict[str, object]], site_ids: set[str], holdout_year: int = HOLDOUT_YEAR) -> tuple[set[str], int]:
    """Return co-usable months. Holdout years are counted as excluded and are not part of the window."""
    months: set[str] = set()
    excluded = 0
    for row in visits:
        if str(row.get("site_id")) not in site_ids:
            continue
        if row.get("observation_status") not in {"reported", "reported_zero"}:
            continue
        year = int(row.get("year") or 0)
        if year >= holdout_year:
            excluded += 1
            continue
        months.add(str(row.get("year_month")))
    return months, excluded


def measurement_applicability(label: object) -> dict[str, object]:
    """A workbook label is not a certified method for every historical month."""
    text = "" if label is None else str(label).strip()
    if not text:
        return {
            "applicability": "no_measurement_information",
            "vocabulary_relation": "absent",
            "document_evidence_for_this_site_year": False,
            "reason": "這個來源檔沒有遊客人次計算方式。",
        }
    exact = [phrase for phrase in GUIDELINE_METHOD_PHRASES if phrase == text or phrase in text]
    relation = "exact_phrase_in_2026_guideline" if exact else "label_not_exact_guideline_phrase"
    return {
        "applicability": "label_only_unchecked",
        "vocabulary_relation": relation,
        "document_evidence_for_this_site_year": False,
        "reason": (
            f"工作簿標籤是「{text}」。現行作業要點第六點自 {GUIDELINE_EFFECTIVE_DATE} 生效，"
            "只列出方法種類，沒有記載這個據點該年使用哪一種，也不能把一年的標籤套到其他月份。"
        ),
    }


def group_measurement_judgment(labels_by_year: dict[int, set[str]]) -> dict[str, str]:
    """Same labels do not make every year fully comparable."""
    present = {year: {label for label in labels if label} for year, labels in labels_by_year.items()}
    present = {year: labels for year, labels in present.items() if labels}
    if not present:
        return {
            "measurement_comparability": "no_measurement_information",
            "reason": "群內據點沒有可追溯的計算方式標籤。",
        }
    flat = {label for labels in present.values() for label in labels}
    mixed_year = any(len(labels) > 1 for labels in present.values())
    if mixed_year or len(flat) > 1:
        return {
            "measurement_comparability": "methods_differ_or_changed",
            "reason": "群內或跨年的計算方式標籤不相同。標籤也還沒有對到據點級文件，所以這些月份不能互相比較。",
        }
    return {
        "measurement_comparability": "label_only_unchecked",
        "reason": "看得到的年份使用同一個工作簿標籤，但作業要點沒有證明這個據點這些月份都用該方法。不能宣布所有年份完全可比。",
    }
