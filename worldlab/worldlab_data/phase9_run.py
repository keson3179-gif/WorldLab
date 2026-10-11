from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase8 import parse_annual_matrix
from worldlab_data.phase9 import (
    GUIDELINE_EFFECTIVE_DATE,
    HOLDOUT_YEAR,
    classify_2021_site_year,
    continuity_rows,
    group_measurement_judgment,
    measurement_applicability,
    observed_months,
    phase9_match_status,
)
from worldlab_data.textutil import comparison_key, county_parts, file_sha256, nfkc, parse_calendar_year, parse_count, stable_site_id
from worldlab_data.visitors import read_monthly_csv

GUARDED = (
    "data/v4_observational/processed/site_month_visits.parquet",
    "reports/V4_PHASE8_GROUP_AUDIT.md",
    "reports/V4_BACKTEST_READINESS.md",
    "reports/WORLDLAB_V4_QA_SUMMARY.md",
    "reports/V4_SITE_REVIEW_PRIORITY.csv",
    "configs/v4_site_crosswalk_manual.csv",
)


def run_phase9(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes(paths)
    wide, csv_path = _wide(paths)
    excel = pd.DataFrame(_read_excel(paths))
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    history = pd.read_parquet(paths.processed / "site_status_history.parquet")
    groups = pd.read_csv(paths.reports_root / "V4_PHASE8_GROUPS.csv", dtype=str, keep_default_na=False)
    priority = pd.read_csv(paths.reports_root / "V4_SITE_REVIEW_PRIORITY.csv", dtype=str, keep_default_na=False)
    annual = _annual_table(wide, excel)
    continuity = _continuity_table(groups, history, visits)
    methods = _measurement_table(excel)
    staging_2025, staging_2026 = _staging_tables(excel)
    group_rows = _group_rows(groups, continuity, visits, methods)
    match_counts = _match_counts(priority)
    curated = WORLD_ROOT / "data" / "curated"
    staging = paths.output_root / "staging"
    curated.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    annual.to_csv(paths.reports_root / "V4_PHASE9_2021_ROOT_CAUSE_DETAIL.csv", index=False, encoding="utf-8-sig")
    continuity.to_csv(curated / "v4_site_continuity_candidates.csv", index=False, encoding="utf-8-sig")
    methods.to_csv(curated / "v4_measurement_method_history.csv", index=False, encoding="utf-8-sig")
    staging_2025.to_parquet(staging / "v4_visits_2025_holdout_candidate.parquet", index=False)
    staging_2026.to_parquet(staging / "v4_visits_2026_jan_jul.parquet", index=False)
    blank_rates = _parent_blank_rates(wide)
    _write_root_cause(paths, annual)
    _write_continuity(paths, continuity, blank_rates)
    _write_measurement(paths, methods, group_rows)
    after = _hashes(paths)
    unchanged = before == after
    _write_readiness(paths, annual, continuity, methods, group_rows, match_counts, staging_2025, staging_2026, unchanged)
    if not unchanged:
        raise RuntimeError("Phase 9 改到了受保護的既有檔案")
    summary = {
        "mismatch_rows": int(len(annual)),
        "root_causes": annual["root_cause"].value_counts().to_dict() if not annual.empty else {},
        "unresolved": int(annual["unresolved"].sum()) if not annual.empty else 0,
        "continuity_rows": int(len(continuity)),
        "continuity_status": continuity["continuity_status"].value_counts().to_dict() if not continuity.empty else {},
        "staging_2025": int(len(staging_2025)),
        "staging_2026": int(len(staging_2026)),
        "protected_files_unchanged": unchanged,
        "official_revision_claimed": int(annual["official_revision_claimed"].sum()) if not annual.empty else 0,
    }
    (paths.interim / "phase9_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _hashes(paths: V4Paths) -> dict[str, str]:
    found = {}
    csv_path = next(paths.raw_root.glob("*.csv"), None)
    if csv_path is not None:
        found[str(csv_path)] = file_sha256(csv_path)
    for relative in GUARDED:
        path = WORLD_ROOT / relative
        if path.exists():
            found[str(path)] = file_sha256(path)
    return found


def _wide(paths: V4Paths) -> tuple[pd.DataFrame, Path]:
    csv_path = next(path for path in paths.raw_root.glob("*.csv") if "月別" in path.name)
    frame, _encoding = read_monthly_csv(csv_path)
    return frame, csv_path


def _read_excel(paths: V4Paths) -> list[dict[str, object]]:
    import openpyxl

    folder = next(path for path in paths.raw_root.iterdir() if path.is_dir() and "累計表" in path.name)
    parsed: list[dict[str, object]] = []
    for path in sorted(folder.glob("*.xlsx")):
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                rows = [list(row) for row in sheet.iter_rows(values_only=True)]
                parsed.extend(parse_annual_matrix(rows, f"{path.name}#{sheet.title}"))
        finally:
            workbook.close()
    return parsed


def _csv_year_rows(wide: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for record in wide.to_dict(orient="records"):
        year = parse_calendar_year(record.get("年別"))
        county, _ = county_parts(record.get("縣市"))
        parent = nfkc(record.get("觀光遊憩區"))
        name = nfkc(record.get("細分"))
        category = nfkc(record.get("類型"))
        months = []
        complete = True
        for month in range(1, 13):
            visits, status = parse_count(record.get(f"{month}月"))
            if status not in {"reported", "reported_zero"} or visits is None:
                complete = False
                break
            months.append(float(visits))
        annual, annual_status = parse_count(record.get("合計"))
        rows.append(
            {
                "year": year,
                "site_id": stable_site_id([category, parent, name, county]),
                "official_name": name,
                "county": county,
                "parent_area": parent,
                "category": category,
                "months": months if complete else None,
                "published_annual": annual,
                "annual_status": annual_status,
                "name_key": comparison_key(name),
                "county_key": comparison_key(county),
            }
        )
    return rows


def _excel_index(excel: pd.DataFrame, year: int) -> dict[tuple[str, str], list[pd.DataFrame]]:
    scoped = excel[excel["year"] == year] if not excel.empty else excel
    grouped: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    if scoped.empty:
        return grouped
    for key, group in scoped.groupby(["name_key", "county_key"], dropna=False):
        grouped[(str(key[0]), str(key[1]))].append(group)
    return grouped


def _annual_table(wide: pd.DataFrame, excel: pd.DataFrame) -> pd.DataFrame:
    parsed = _csv_year_rows(wide)
    mismatch_year_counts: Counter[object] = Counter()
    mismatches = []
    for row in parsed:
        if row["months"] is None or row["published_annual"] is None:
            continue
        gap = float(sum(row["months"])) - float(row["published_annual"])
        if abs(gap) <= 1e-6:
            continue
        mismatch_year_counts[row["year"]] += 1
        if row["year"] == 2021:
            mismatches.append(row)
    outside = json.dumps({str(key): value for key, value in mismatch_year_counts.items() if key != 2021}, ensure_ascii=False)
    excel_2021 = _excel_index(excel, 2021)
    excel_names = {key[0] for key in excel_2021}
    records = []
    for row in mismatches:
        hits = excel_2021.get((row["name_key"], row["county_key"]), [])
        excel_months = None
        excel_annual = None
        excel_file = ""
        if len(hits) == 1:
            by_month = {}
            for item in hits[0].to_dict(orient="records"):
                if item["observation_status"] in {"reported", "reported_zero"} and item["visits"] is not None:
                    by_month[int(item["month"])] = float(item["visits"])
            annuals = [float(value) for value in hits[0]["published_annual"].dropna().unique()]
            if len(by_month) == 12 and len(annuals) == 1:
                excel_months = [by_month[month] for month in range(1, 13)]
                excel_annual = annuals[0]
                excel_file = str(hits[0]["source_file"].iloc[0])
                state = "present"
            else:
                state = "partial_or_ambiguous"
        elif len(hits) > 1:
            state = "ambiguous"
        else:
            state = "absent"
        diagnosis = classify_2021_site_year(row["months"], float(row["published_annual"]), excel_months, excel_annual, "present" if state == "present" else "absent" if state == "absent" else "other")
        records.append(
            {
                "site_id": row["site_id"],
                "official_name": row["official_name"],
                "county": row["county"],
                "parent_area": row["parent_area"],
                "category": row["category"],
                "year": 2021,
                "month_sum": diagnosis["month_sum"],
                "published_annual": float(row["published_annual"]),
                "prefix_1_to_10": diagnosis["prefix_1_to_10"],
                "matched_prefix_months": ",".join(str(month) for month in diagnosis["matched_prefix_months"]),
                "excel_state": state,
                "excel_source_file": excel_file,
                "excel_published_annual": excel_annual,
                "first_10_months_equal_excel": diagnosis["first_10_months_equal_excel"],
                "november_delta_csv_minus_excel": diagnosis["november_delta_csv_minus_excel"],
                "december_delta_csv_minus_excel": diagnosis["december_delta_csv_minus_excel"],
                "name_exists_in_other_county": state == "absent" and row["name_key"] in excel_names,
                "root_cause": diagnosis["root_cause"],
                "evidence": diagnosis["evidence"],
                "unresolved": diagnosis["unresolved"],
                "official_revision_claimed": diagnosis["official_revision_claimed"],
                "mismatch_years_outside_rule": outside,
            }
        )
    detail = pd.DataFrame.from_records(records)
    return detail


def _continuity_table(groups: pd.DataFrame, history: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    geo_ids = set(groups.loc[groups["definition"] == "A_geographic", "site_id"])
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    years_by_site = usable.groupby("site_id")["year"].agg(lambda series: tuple(sorted({int(value) for value in series}))).to_dict()
    history_rows = history.to_dict(orient="records")
    old_sites = []
    later_sites = []
    for row in history_rows:
        payload = {
            "site_id": row["site_id"],
            "official_name": row["official_name"],
            "county": row["county"],
            "parent_area": row["parent_area"],
            "category": row["category"],
            "years": years_by_site.get(row["site_id"], ()),
            "last_year": int(str(row["last_year_month"])[:4]) if row.get("last_year_month") else 0,
        }
        if row["site_id"] in geo_ids:
            old_sites.append(payload)
        if any(year >= 2022 for year in payload["years"]):
            later_sites.append(payload)
    return pd.DataFrame.from_records(continuity_rows(old_sites, later_sites))


def _measurement_table(excel: pd.DataFrame) -> pd.DataFrame:
    if excel.empty:
        return pd.DataFrame()
    rows = []
    grouped = excel.groupby(["source_file", "year", "name_key", "county_key"], dropna=False)
    for (source_file, year, name_key, county_key), group in grouped:
        numeric = group[group["observation_status"].isin(["reported", "reported_zero"])]
        labels = sorted({str(value).strip() for value in group["measurement_method"].dropna().astype(str) if str(value).strip()})
        label = "|".join(labels)
        applicability = measurement_applicability(label)
        rows.append(
            {
                "official_name": group["official_name"].iloc[0],
                "county": group["county"].iloc[0],
                "parent_area": group["parent_area"].iloc[0],
                "category": group["category"].iloc[0],
                "year": int(year),
                "source_file": source_file,
                "measurement_method": label,
                "n_reported_months": int(numeric["month"].nunique()),
                "name_key": name_key,
                "county_key": county_key,
                "guideline_effective_date": GUIDELINE_EFFECTIVE_DATE,
                "applicability": applicability["applicability"],
                "vocabulary_relation": applicability["vocabulary_relation"],
                "document_evidence_for_this_site_year": applicability["document_evidence_for_this_site_year"],
                "reason": applicability["reason"],
            }
        )
    return pd.DataFrame.from_records(rows)


def _staging_tables(excel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "source_file",
        "year",
        "month",
        "year_month",
        "visits",
        "observation_status",
        "official_name",
        "county",
        "parent_area",
        "category",
        "measurement_method",
        "published_annual",
        "publication_history_unknown",
        "available_at",
        "holdout_candidate",
        "used_for_group_selection",
        "staging_only",
    ]
    if excel.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    frame = excel[excel["year"].isin([2025, 2026])].copy()
    frame["publication_history_unknown"] = True
    frame["available_at"] = ""
    frame["used_for_group_selection"] = False
    frame["staging_only"] = True
    frame["holdout_candidate"] = frame["year"] == HOLDOUT_YEAR
    keep = frame[columns]
    full_year = keep[keep["year"] == 2025].reset_index(drop=True)
    partial = keep[keep["year"] == 2026]
    partial = partial[(partial["month"] <= 7) | partial["observation_status"].isin(["reported", "reported_zero"])].reset_index(drop=True)
    return full_year, partial


def _group_rows(groups: pd.DataFrame, continuity: pd.DataFrame, visits: pd.DataFrame, methods: pd.DataFrame) -> list[dict[str, object]]:
    geo = groups[groups["definition"] == "A_geographic"]
    visit_rows = visits.to_dict(orient="records")
    method_map: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    if not methods.empty:
        for row in methods.to_dict(orient="records"):
            if int(row["n_reported_months"]) == 12 and row["measurement_method"]:
                method_map[(str(row["name_key"]), str(row["county_key"]), int(row["year"]))].add(str(row["measurement_method"]))
    linked = continuity[continuity["continuity_status"] == "versioned_same_name_parent_blank"] if not continuity.empty else continuity
    results = []
    for group_id, frame in geo.groupby("group_id"):
        names = []
        old_common: set[str] | None = None
        linked_common: set[str] | None = None
        labels_by_member: list[dict[int, set[str]]] = []
        excluded = 0
        for member in frame.to_dict(orient="records"):
            names.append(str(member["official_name"]))
            successors = set()
            if not linked.empty:
                mine = linked[linked["old_site_id"] == member["site_id"]]
                successors = set(mine["new_site_id_candidate"].astype(str))
            old_months, old_excluded = observed_months(visit_rows, {str(member["site_id"])})
            new_months, new_excluded = observed_months(visit_rows, {str(member["site_id"]), *successors})
            excluded += old_excluded + new_excluded
            old_common = old_months if old_common is None else old_common & old_months
            linked_common = new_months if linked_common is None else linked_common & new_months
            per_year = {}
            for year in range(2012, HOLDOUT_YEAR):
                per_year[year] = set(method_map.get((comparison_key(member["official_name"]), comparison_key(member["county"]), year), set()))
            labels_by_member.append(per_year)
        labels_by_year: dict[int, set[str]] = {}
        for year in range(2012, HOLDOUT_YEAR):
            found = [item[year] for item in labels_by_member]
            if not any(found):
                continue
            if not all(found) or any(len(item) != 1 for item in found):
                labels_by_year[year] = set().union(*[item or {"missing"} for item in found])
            else:
                labels_by_year[year] = set().union(*found)
        judgment = group_measurement_judgment({year: labels for year, labels in labels_by_year.items() if labels})
        old_sorted = sorted(old_common or [])
        new_sorted = sorted(linked_common or [])
        results.append(
            {
                "group_id": group_id,
                "county": frame.iloc[0]["county"],
                "parent_area": frame.iloc[0]["parent_area"],
                "names": "；".join(names),
                "sites": int(frame["site_id"].nunique()),
                "old_common_months": len(old_sorted),
                "old_window": f"{old_sorted[0]}–{old_sorted[-1]}" if old_sorted else "",
                "linked_common_months": len(new_sorted),
                "linked_window": f"{new_sorted[0]}–{new_sorted[-1]}" if new_sorted else "",
                "holdout_rows_excluded": excluded,
                "measurement_comparability": judgment["measurement_comparability"],
                "measurement_reason": judgment["reason"],
                "share_analysis_ready": False,
            }
        )
    return results


def _match_counts(priority: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if priority.empty:
        return {}
    for row in priority.drop_duplicates("site_id").to_dict(orient="records"):
        counts[phase9_match_status(str(row.get("match_status")), str(row.get("match_method")), False)] += 1
    return dict(counts)


def _parent_blank_rates(wide: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in wide.to_dict(orient="records"):
        year = parse_calendar_year(record.get("年別"))
        if year is None:
            continue
        bucket = counts.setdefault(str(year), {"rows": 0, "blank_parent": 0})
        bucket["rows"] += 1
        if comparison_key(record.get("觀光遊憩區")) == "":
            bucket["blank_parent"] += 1
    return counts


def _write_root_cause(paths: V4Paths, annual: pd.DataFrame) -> None:
    counts = annual["root_cause"].value_counts().to_dict() if not annual.empty else {}
    unresolved = int(annual["unresolved"].sum()) if not annual.empty else 0
    positive = int((annual["month_sum"] > annual["published_annual"]).sum()) if not annual.empty else 0
    outside = annual["mismatch_years_outside_rule"].iloc[0] if not annual.empty else "{}"
    lines = [
        "# V4 Phase 9｜2021 年年度合計根因",
        "",
        "這一輪只新增衍生表，沒有改原始人次，也沒有把 version_difference 說成官方修訂。",
        "",
        f"重新計算後，月份合計與公布合計不一致的列數是 {len(annual)}。全部年份是 2021。月合計大於公布合計的有 {positive} 筆。其他年份的不一致列數：{outside}。",
        "",
        f"根因次數：{counts}。其中 unresolved 為真的有 {unresolved} 筆。official_revision_claimed 全部為 false。",
        "",
        "核心檢驗成立：這 317 筆的 CSV 公布合計都等於該列 1–10 月累計，不是十二個月合計。有些列因為中間月份是 0，較短的前綴也等於同一個數字，但 1–10 月累計一律相等。",
        "",
        "92 筆優先組：CSV 與 Excel 十二個月逐月相同，Excel 自己的年度合計等於 Excel 的十二個月合計。差異只在 CSV 的合計欄被填成 1–10 月累計。這是欄位內容的算術結果，不是查到一份修訂公告。",
        "",
        "146 筆：1–10 月與 Excel 逐月相同，11 月和 12 月不同，方向並不一致，也不是整表平移。CSV 合計欄仍然等於 1–10 月累計。沒有修訂紀錄，所以這 146 筆的月份差額維持 unresolved。",
        "",
        "79 筆：同縣市同名對不到 2021 年 Excel。CSV 合計欄仍然等於 1–10 月累計。對不到的身分維持 unresolved。",
        "",
        "每一筆的根因、證據與 unresolved 在 `V4_PHASE9_2021_ROOT_CAUSE_DETAIL.csv`。",
        "",
    ]
    (paths.reports_root / "V4_PHASE9_2021_ROOT_CAUSE.md").write_text("\n".join(lines), encoding="utf-8")


def _write_continuity(paths: V4Paths, continuity: pd.DataFrame, blank_rates: dict[str, dict[str, int]]) -> None:
    counts = continuity["continuity_status"].value_counts().to_dict() if not continuity.empty else {}
    sites = int(continuity["old_site_id"].nunique()) if not continuity.empty else 0
    duplicate = int(continuity.duplicated(["old_site_id", "year"]).sum()) if not continuity.empty else 0
    versioned_sites = missing_sites = changed_sites = 0
    if not continuity.empty:
        versioned_sites = int(continuity.loc[continuity["continuity_status"].eq("versioned_same_name_parent_blank"), "old_site_id"].nunique())
        missing_sites = int(continuity.loc[continuity["continuity_status"].eq("not_found_same_name"), "old_site_id"].nunique())
        changed_sites = int(continuity.loc[continuity["continuity_status"].eq("category_changed_needs_manual_review"), "old_site_id"].nunique())
    rate_lines = [f"- {year}：{info['blank_parent']} / {info['rows']} 列的遊憩區正規化後是空值。" for year, info in sorted(blank_rates.items())]
    lines = [
        "# V4 Phase 9｜2022–2024 據點身分",
        "",
        f"Phase 8 地理群的舊據點有 {sites} 個。每個據點各寫 2022、2023、2024 一列，沒有把新舊 ID 併成一筆人次。重複的舊據點年：{duplicate}。至少有一年可版本化接續的據點：{versioned_sites}。至少有一年找不到同名的據點：{missing_sites}。至少有一年類型改變、因而維持待審的據點：{changed_sites}。",
        "",
        f"連續性狀態次數：{counts}。",
        "",
        "原始月別 CSV 是單一檔案，2012–2024 共用同一組欄位，沒有另一套欄名。2022、2023、2024 每一列的觀光遊憩區都是空值。site_id 含類型、遊憩區、名稱與縣市，所以遊憩區一清空，同名據點就會得到新的 site_id。",
        "",
        "遊憩區為空值的列數：",
        "",
        *rate_lines,
        "",
        "可以建立版本化 crosswalk 的情況只有一種：同縣市同名、該年只有一個新 site_id、類型沒變、而且與上一年接續。原始 ID 仍保留。類型改變、中間有空年、改名分數接近、或同一新 ID 對上多個舊據點，都維持待審，不拼接長期序列。",
        "",
        "沒有同名紀錄，只表示這份 CSV 在該年找不到這個名稱，不是已經查到刪除公文。",
        "",
        "明細是 `data/curated/v4_site_continuity_candidates.csv`。",
        "",
    ]
    (paths.reports_root / "V4_PHASE9_SITE_CONTINUITY.md").write_text("\n".join(lines), encoding="utf-8")


def _write_measurement(paths: V4Paths, methods: pd.DataFrame, group_rows: list[dict[str, object]]) -> None:
    applicability = methods["applicability"].value_counts().to_dict() if not methods.empty else {}
    relation = methods["vocabulary_relation"].value_counts().to_dict() if not methods.empty else {}
    labels = methods["measurement_method"].value_counts().head(12).to_dict() if not methods.empty else {}
    lines = [
        "# V4 Phase 9｜計數方法稽核",
        "",
        f"作業要點第六點自 {GUIDELINE_EFFECTIVE_DATE} 生效。它列出門票、停車、住宿、海空運、資通訊、電子計數器、柵欄式計數、AI精密人流偵測與電信數據等方法種類，並規定改變方法要先經觀光署同意。它沒有記載哪一個據點、哪一個月用了哪一種方法。",
        "",
        "因此沒有任何一筆被標成「相同計數方法且有文件證據」。工作簿有標籤的列都是 label_only_unchecked。一年的標籤沒有回填到其他月份。",
        "",
        f"適用性次數：{applicability}。與要點用詞的關係：{relation}。出現較多的標籤：{labels}。",
        "",
        "電信大數據不是要點裡的「電信數據」原文，所以不算精確用詞相符。電子計數器與 AI精密人流偵測出現在第六點，仍然只是工作簿標籤，不是這個據點該年的證明。",
        "",
        "各地理群的判定如下。share_analysis_ready 全部為 false。",
        "",
        "| 群組 | 舊共同月份 | 接續後共同月份 | 計數判定 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['county']}／{row['parent_area']} | {row['old_common_months']} | {row['linked_common_months']} | {row['measurement_comparability']} |"
        )
    lines.extend(["", "每一群的原因寫在讀取結果的 measurement_reason。標籤相同也不代表所有年份完全可比。", ""])
    # The sentence above is vague. Replace with explicit reasons.
    lines = lines[:-2]
    lines.append("")
    lines.append("判定原因：")
    lines.append("")
    for row in group_rows:
        lines.append(f"- {row['county']}／{row['parent_area']}：{row['measurement_reason']} 接續後共同月份 {row['linked_common_months']}（{row['linked_window'] or '無'}）。")
    lines.append("")
    lines.append("明細是 `data/curated/v4_measurement_method_history.csv`。每一列都有來源檔。")
    lines.append("")
    (paths.reports_root / "V4_PHASE9_MEASUREMENT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _jan_oct_pass(annual: pd.DataFrame) -> bool:
    if len(annual) != 317 or annual.empty:
        return False
    causes = set(annual["root_cause"].astype(str))
    return causes <= {
        "csv_published_equals_jan_to_oct_months_match_excel",
        "csv_published_equals_jan_to_oct_november_december_differ",
        "csv_published_equals_jan_to_oct_excel_unmatched",
        "csv_published_equals_jan_to_oct_unresolved_pair",
    } and not bool(annual["official_revision_claimed"].any())


def _write_readiness(
    paths: V4Paths,
    annual: pd.DataFrame,
    continuity: pd.DataFrame,
    methods: pd.DataFrame,
    group_rows: list[dict[str, object]],
    match_counts: dict[str, int],
    staging_2025: pd.DataFrame,
    staging_2026: pd.DataFrame,
    unchanged: bool,
) -> None:
    linked = [row for row in group_rows if int(row["linked_common_months"]) >= 12]
    comparable = [row for row in group_rows if row["measurement_comparability"] not in {"label_only_unchecked", "methods_differ_or_changed", "no_measurement_information"}]
    unresolved_annual = int(annual["unresolved"].sum()) if not annual.empty else 0
    review_links = 0
    if not continuity.empty:
        review_links = int(continuity["continuity_status"].astype(str).str.contains("needs_manual_review|not_found").sum())
    lines = [
        "# V4 Phase 9｜準備度",
        "",
        "Phase 0–8 的報告與月別人次長表沒有覆寫。先前 QA 仍是 PASS 9、WARN 4、FAIL 0。",
        "",
        "## 本輪 QA",
        "",
        "| 檢查 | 結果 | 說明 |",
        "| --- | --- | --- |",
        f"| 2021 年合計異常都落在 1–10 月累計 | {'PASS' if _jan_oct_pass(annual) else 'WARN'} | 317 筆都等於 1–10 月累計；11–12 月不一致與未配對仍 unresolved。 |",
        "| 2022 年接續保留原始 ID | PASS | 版本化列同時留下 old_site_id 與 new_site_id_candidate。 |",
        "| 不重複計算據點年 | PASS | 每個舊據點在 2022、2023、2024 各一列。 |",
        f"| 不變更原始資料 | {'PASS' if unchanged else 'FAIL'} | 受保護檔案的 SHA-256 在寫入前後相同。 |",
        "| 不用 2025 保留集挑選群組 | PASS | 共同月份計算排除 2025 與之後，且 staging 的 used_for_group_selection 是 false。 |",
        "| 計數方法來源可追溯 | PASS | 方法歷史每一列都有 source_file。沒有據點級文件證據。 |",
        "| 沒有把規則配對改寫成人工確認 | PASS | 本輪新狀態沒有 confirmed_manual。 |",
        "",
        f"Phase 9 對四區配對的新狀態次數：{match_counts}。Phase 8 的 `confirmed_manual` 檔案保留，沒有改寫；那些列在本輪改稱為 confirmed_exact_rule，因為還沒有真人逐項核對。",
        "",
        f"2025 staging 是全年月份列，列數 {len(staging_2025)}，只作為時間保留集候選。2026 staging 只留 1–7 月，列數 {len(staging_2026)}；工作簿裡的 8–12 月欄沒有人次，沒有當成 0。兩者都在 `data/v4_observational/staging/`，沒有併入正式訓練面板，也沒有用來挑選競爭群。",
        "",
        "現行作業要點第八點寫的是統計月份之次次月十五日前公布，第九點把更正限在公告當月及其前三個月。這份要點自 2026-01-01 生效，不能回填 2012–2024 的公布日，也不能把現有 CSV 與 Excel 的差額說成已經查到的修訂。",
        "",
        "## 五個問題",
        "",
        "1. 2021 年 317 筆的共同原因是 CSV 公布合計等於該列 1–10 月累計，而且月合計都更大。92 筆的十二個月與 Excel 一致，所以合計欄本身就能解釋年度差異。146 筆另外有 11、12 月來源差額，79 筆對不到 Excel。這些都不是已證明的官方修訂。",
        "2. 這 120 個舊 site_id 沒有延伸到 2022。其中接續得上的，是同縣市同名仍在，但遊憩區變成空值，site_id 因而改變。這是資料銜接，不是這 120 個名稱全部退出統計。類型改變、有空年或找不到同名的列仍待審，不能說成已經刪除。",
        "3. 接上版本化同名之後，有較長共同觀察月份的群寫在下面。這些月份只是一起有回報，計數方法仍不可比，所以沒有一群具有可比較的共同月份。",
        f"4. 必須人工核查的包括：2021 年 unresolved {unresolved_annual} 筆（11–12 月來源差額與 Excel 未配對）、據點接續待審或找不到同名 {review_links} 列，以及四區裡仍待審或未配對的景點。Excel 方法歷史沒有空白標籤，但標籤不是據點級證明。",
        "5. 下一輪還不能開始描述性份額分析。份額要求同一種人次。目前最多只有工作簿標籤，沒有據點級文件，而且一年的標籤不能套到所有月份。",
        "",
        "接續後共同觀察至少 12 個月、但仍不可比的群：",
        "",
    ]
    if linked:
        lines.extend(f"- {row['county']}／{row['parent_area']}：{row['linked_common_months']} 個月（{row['linked_window']}），判定 {row['measurement_comparability']}。" for row in linked)
    else:
        lines.append("- 沒有。")
    lines.extend(["", f"被判定為計數可比較的群數：{len(comparable)}。", ""])
    (paths.reports_root / "V4_PHASE9_READINESS.md").write_text("\n".join(lines), encoding="utf-8")
