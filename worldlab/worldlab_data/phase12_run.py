from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from worldlab_data.config import WORLD_ROOT, V4Paths, load_paths
from worldlab_data.phase10 import parse_annual_sheet
from worldlab_data.phase11 import retrospective_origin_count
from worldlab_data.phase12 import (
    alignment_pattern,
    classify_category_discrepancy,
    confirmed_parser_error,
    link_decision,
    strict_asof_eligible,
    valid_share_vector,
)
from worldlab_data.textutil import comparison_key, county_parts, file_sha256, nfkc, parse_count, stable_site_id
from worldlab_data.visitors import read_monthly_csv

GUARDED = (
    "data/v4_observational/processed/site_month_visits.parquet",
    "data/curated/v4_canonical_site_crosswalk.csv",
    "data/curated/v4_exploratory_visit_shares_conditional.csv",
    "data/curated/v4_monthly_concentration_conditional.parquet",
    "reports/V4_PHASE11_CONCENTRATION_EXPLORATION.md",
    "reports/V4_PHASE10_2021_MONTH_EVIDENCE.csv",
)


def run_phase12(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    before = _hashes()
    curated = WORLD_ROOT / "data" / "curated"
    groups = pd.read_csv(paths.reports_root / "V4_PHASE8_GROUPS.csv", dtype=str, keep_default_na=False)
    comparability = pd.read_csv(curated / "v4_group_comparability.csv", dtype=str, keep_default_na=False)
    crosswalk = pd.read_csv(curated / "v4_canonical_site_crosswalk.csv", dtype=str, keep_default_na=False)
    evidence = pd.read_csv(paths.reports_root / "V4_PHASE10_2021_MONTH_EVIDENCE.csv", dtype=str, keep_default_na=False)
    shares = pd.read_csv(curated / "v4_exploratory_visit_shares_conditional.csv", dtype=str, keep_default_na=False)
    csv_rows = _csv_rows(paths)
    workbooks = _workbooks(paths)
    identity = _identity_rows(groups, comparability, crosswalk, csv_rows, workbooks)
    alignment = _alignment_rows(evidence, workbooks.get(2021, []))
    panel = _panel(shares, comparability)
    if any(int(str(row["year_month"])[:4]) >= 2025 for row in panel):
        raise RuntimeError("2025 entered the forecast panel")
    if any(row["strict_asof_eligible"] for row in panel):
        raise RuntimeError("strict as-of was marked without a publication date")
    curated.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(identity).to_csv(curated / "v4_2024_identity_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(alignment).to_csv(curated / "v4_2021_alignment_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(panel).to_parquet(curated / "v4_forecast_panel_candidate.parquet", index=False)
    holdout = _holdout_flag(paths)
    _write_reports(paths, identity, alignment, panel, holdout)
    payload = {
        "analysis_role": "forensic_audit",
        "identity_rows": len(identity),
        "safe_links": int(sum(row["safe_to_link"] for row in identity)),
        "category_reasons": {str(key): int(value) for key, value in pd.Series([row["category_change_reason"] for row in identity]).value_counts().items()},
        "parser_errors": int(sum(row["parser_error"] for row in alignment)),
        "previous_row_pattern": int(sum(row["pattern"] == "equals_previous_visit_row" for row in alignment)),
        "panel_rows": len(panel),
        "holdout_visits_used": False,
        "model_fitted": False,
        "protected_files_unchanged": _hashes() == before,
    }
    (paths.interim / "phase12_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not payload["protected_files_unchanged"]:
        raise RuntimeError("phase 12 changed a protected file")
    return payload


def _csv_rows(paths: V4Paths) -> list[dict[str, object]]:
    csv_path = next(path for path in paths.raw_root.glob("*.csv") if "月別" in path.name)
    frame, _encoding = read_monthly_csv(csv_path)
    rows = []
    for index, record in enumerate(frame.to_dict(orient="records"), start=2):
        year_text = nfkc(record.get("年別"))
        if year_text not in {"2023", "2024"}:
            continue
        county, _english = county_parts(record.get("縣市"))
        months = {}
        for month in range(1, 13):
            value, status = parse_count(record.get(f"{month}月"))
            if status in {"reported", "reported_zero"} and value is not None:
                months[month] = float(value)
        category = nfkc(record.get("類型"))
        parent = nfkc(record.get("觀光遊憩區"))
        name = nfkc(record.get("細分"))
        rows.append(
            {
                "csv_row": index,
                "year": int(year_text),
                "category": category,
                "parent": parent,
                "official_name": name,
                "county": county,
                "months": months,
                "raw_site_id": stable_site_id([category, parent, name, county]),
                "name_key": comparison_key(name),
                "county_key": comparison_key(county),
            }
        )
    return rows


def _workbooks(paths: V4Paths) -> dict[int, list[dict[str, object]]]:
    folder = next(path for path in paths.raw_root.iterdir() if path.is_dir() and "累計表" in path.name)
    import openpyxl

    found = {}
    for year, token in ((2021, "110年"), (2023, "112年"), (2024, "113年")):
        path = next(folder.glob(f"*{token}*.xlsx"))
        workbook = openpyxl.load_workbook(path, read_only=False, data_only=True)
        sheet = workbook.worksheets[0]
        matrix = [list(row) for row in sheet.iter_rows(values_only=True)]
        merges = [(item.min_row, item.max_row, item.min_col, item.max_col) for item in sheet.merged_cells.ranges]
        month_merges = {(min_row, max_row, min_col, max_col) for min_row, max_row, min_col, max_col in merges if min_col != 1}
        parsed = parse_annual_sheet(matrix, merges, path.name, sheet.title)
        by_row = {int(row["source_row_number"]): row for row in parsed}
        ordered = []
        for row in sorted(parsed, key=lambda item: int(item["source_row_number"])):
            excel_row = int(row["source_row_number"])
            ordered.append(
                {
                    **row,
                    "november": row["months"].get(11),
                    "december": row["months"].get(12),
                    "month_cell_merged": any(min_row <= excel_row <= max_row for min_row, max_row, _min_col, _max_col in month_merges),
                    "worksheet_november": row["months"].get(11),
                    "worksheet_december": row["months"].get(12),
                }
            )
        for index, row in enumerate(ordered):
            row["previous_name"] = ordered[index - 1]["official_name"] if index else ""
            row["next_name"] = ordered[index + 1]["official_name"] if index + 1 < len(ordered) else ""
            row["previous_november"] = ordered[index - 1]["november"] if index else None
            row["previous_december"] = ordered[index - 1]["december"] if index else None
            row["next_november"] = ordered[index + 1]["november"] if index + 1 < len(ordered) else None
            row["next_december"] = ordered[index + 1]["december"] if index + 1 < len(ordered) else None
            row["row_gap_before"] = int(row["source_row_number"]) - int(ordered[index - 1]["source_row_number"]) if index else 0
        workbook.close()
        found[year] = ordered
        _ = by_row
    return found


def _identity_rows(groups, comparability, crosswalk, csv_rows, workbooks) -> list[dict[str, object]]:
    eligible = set(
        comparability.loc[
            comparability["definition"].eq("A_geographic") & comparability["exploratory_share_eligible"].str.lower().eq("true"),
            "group_id",
        ]
    )
    members = groups[groups["definition"].eq("A_geographic") & groups["group_id"].isin(eligible)]
    by_raw = {str(row["raw_site_id"]): row for row in crosswalk.to_dict(orient="records")}
    csv_index: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in csv_rows:
        csv_index[(str(row["name_key"]), str(row["county_key"]), int(row["year"]))].append(row)
    excel_index: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for year, rows in workbooks.items():
        if year not in {2023, 2024}:
            continue
        for row in rows:
            excel_index[(comparison_key(row["official_name"]), comparison_key(row["county"]), year)].append(row)
    output = []
    for member in members.to_dict(orient="records"):
        anchor = by_raw.get(str(member["site_id"]), {})
        uid = str(anchor.get("site_uid") or member["site_id"])
        chain = [row for row in crosswalk.to_dict(orient="records") if str(row["site_uid"]) == uid]
        row_2023 = next((row for row in chain if int(row["first_year"] or 0) <= 2023 <= int(row["last_year"] or 0)), None)
        name = str((row_2023 or anchor).get("official_name") or member["official_name"])
        county = str((row_2023 or anchor).get("county") or member["county"])
        category_2023 = str((row_2023 or anchor).get("category") or "")
        raw_2023 = str((row_2023 or anchor).get("raw_site_id") or member["site_id"])
        csv_2023 = _only(csv_index.get((comparison_key(name), comparison_key(county), 2023), []))
        csv_2024_hits = csv_index.get((comparison_key(name), comparison_key(county), 2024), [])
        csv_2024 = csv_2024_hits[0] if len(csv_2024_hits) == 1 else None
        excel_2024_hits = excel_index.get((comparison_key(name), comparison_key(county), 2024), [])
        excel_2024 = excel_2024_hits[0] if len(excel_2024_hits) == 1 else None
        months_match = _months_match(csv_2024, excel_2024)
        same_category = bool(csv_2024) and comparison_key(category_2023) == comparison_key(csv_2024["category"])
        decision = link_decision(
            found_2024=bool(csv_2024_hits),
            same_name=bool(csv_2024),
            same_county=bool(csv_2024),
            same_category=same_category,
            unique_candidate=len(csv_2024_hits) == 1,
            months_match_excel=months_match,
        )
        if same_category and months_match and excel_2024 and comparison_key(excel_2024["category"]) != comparison_key(csv_2024["category"]):
            decision = {
                **decision,
                "identity_status": "rule_inferred",
                "continuity_evidence": "CSV 類型跨年未變，12 個月與 Excel 同名列一致，但 Excel 合併類型用字和 CSV 不同。",
            }
        if csv_2024 is None:
            reason = "not_found"
        elif comparison_key(category_2023) == comparison_key(csv_2024["category"]):
            reason = "unchanged"
        else:
            reason = classify_category_discrepancy(
                str(csv_2024["category"]),
                "" if excel_2024 is None else str(excel_2024["category"]),
                "" if excel_2024 is None else str(excel_2024["raw_category_cell"]),
                filled_by_parser=False,
            )
        output.append(
            {
                "group_id": member["group_id"],
                "county": member["county"],
                "parent_area": member["parent_area"],
                "official_name": name,
                "canonical_site_uid": uid,
                "raw_site_id_2023": raw_2023,
                "raw_site_id_2024": "" if csv_2024 is None else csv_2024["raw_site_id"],
                "csv_row_2023": "" if csv_2023 is None else csv_2023["csv_row"],
                "csv_row_2024": "" if csv_2024 is None else csv_2024["csv_row"],
                "csv_category_2023": category_2023,
                "csv_category_2024": "" if csv_2024 is None else csv_2024["category"],
                "excel_row_2024": "" if excel_2024 is None else excel_2024["source_row_number"],
                "excel_raw_category_cell_2024": "" if excel_2024 is None else excel_2024["raw_category_cell"],
                "excel_category_2024": "" if excel_2024 is None else excel_2024["category"],
                "months_match_excel_2024": months_match,
                "continuity_evidence": decision["continuity_evidence"],
                "category_change_reason": reason if csv_2024 is not None or reason == "not_found" else decision["category_change_reason"],
                "identity_status": decision["identity_status"],
                "safe_to_link": bool(decision["safe_to_link"]),
                "human_reviewed": False,
            }
        )
    return output


def _alignment_rows(evidence: pd.DataFrame, excel_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in excel_rows:
        index[(comparison_key(row["official_name"]), comparison_key(row["county"]))] = row
    output = []
    scoped = evidence[evidence["source_class"].eq("B_november_december_conflict")]
    for record in scoped.to_dict(orient="records"):
        excel = index.get((comparison_key(record["official_name"]), comparison_key(record["county"])))
        csv_nov = float(record["csv_november"])
        csv_dec = float(record["csv_december"])
        if excel is None:
            output.append(
                {
                    "official_name": record["official_name"],
                    "county": record["county"],
                    "raw_site_id": record["raw_site_id"],
                    "excel_row": "",
                    "pattern": "excel_name_unmatched",
                    "parser_error": False,
                    "chosen_version": "",
                    "csv_november": csv_nov,
                    "csv_december": csv_dec,
                    "excel_november": "",
                    "excel_december": "",
                    "previous_name": "",
                    "next_name": "",
                    "month_cell_merged": False,
                    "blank_row_before": False,
                    "note": "Excel 沒有同名同縣市的列。數值相同不會被拿來配對。",
                }
            )
            continue
        excel_nov = float(excel["november"])
        excel_dec = float(excel["december"])
        nov_pattern = alignment_pattern(csv_nov, excel_nov, _float_or_none(excel["previous_november"]), _float_or_none(excel["next_november"]))
        dec_pattern = alignment_pattern(csv_dec, excel_dec, _float_or_none(excel["previous_december"]), _float_or_none(excel["next_december"]))
        pattern = nov_pattern if nov_pattern == dec_pattern else "mixed"
        parser_error = confirmed_parser_error(excel["worksheet_november"], excel_nov) or confirmed_parser_error(excel["worksheet_december"], excel_dec)
        output.append(
            {
                "official_name": record["official_name"],
                "county": record["county"],
                "raw_site_id": record["raw_site_id"],
                "excel_row": excel["source_row_number"],
                "pattern": pattern,
                "november_pattern": nov_pattern,
                "december_pattern": dec_pattern,
                "parser_error": parser_error,
                "chosen_version": "",
                "csv_november": csv_nov,
                "csv_december": csv_dec,
                "excel_november": excel_nov,
                "excel_december": excel_dec,
                "previous_name": excel["previous_name"],
                "next_name": excel["next_name"],
                "month_cell_merged": bool(excel["month_cell_merged"]),
                "blank_row_before": int(excel["row_gap_before"]) > 1,
                "note": "名稱列自己的儲存格與解析值一致。相鄰列數值相同只是觀察到的模式，沒有改寫人次。",
            }
        )
    return output


def _panel(shares: pd.DataFrame, comparability: pd.DataFrame) -> list[dict[str, object]]:
    eligible = set(
        comparability.loc[
            comparability["definition"].eq("A_geographic") & comparability["exploratory_share_eligible"].str.lower().eq("true"),
            "group_id",
        ]
    )
    rows = []
    scoped = shares[shares["group_id"].isin(eligible) & shares["year_month"].str.slice(0, 4).astype(int).lt(2024)]
    for record in scoped.to_dict(orient="records"):
        published_at = ""
        rows.append(
            {
                "group_id": record["group_id"],
                "site_uid": record["site_uid"],
                "year_month": record["year_month"],
                "visits": float(record["visits"]),
                "share": float(record["share"]),
                "dataset_role": "exploratory_retrospective",
                "confirmatory_holdout_candidate": False,
                "strict_asof_eligible": strict_asof_eligible(published_at),
                "publication_history_unknown": True,
                "publication_delay_assumption": "公布日無法重建。探索性回測只使用已入庫的歷史月份，不把延遲當成已知。",
                "target": "reported_visit_distribution",
                "counting_limits": record["counting_limits"],
                "used_for_model_selection": False,
            }
        )
    return rows


def _holdout_flag(paths: V4Paths) -> dict[str, object]:
    staging = paths.output_root / "staging"
    files = sorted(staging.glob("v4_visits_202*.parquet")) if staging.exists() else []
    years = []
    flags = []
    count = 0
    for path in files:
        frame = pd.read_parquet(path, columns=["year", "used_for_group_selection"])
        count += int(len(frame))
        years.extend(int(value) for value in frame["year"].dropna().unique())
        flags.extend(bool(value) for value in frame["used_for_group_selection"].dropna().unique())
    return {"rows": count, "years": sorted(set(years)), "used_for_group_selection": any(flags)}


def _write_reports(paths: V4Paths, identity: list[dict[str, object]], alignment: list[dict[str, object]], panel: list[dict[str, object]], holdout: dict[str, object]) -> None:
    (paths.reports_root / "V4_PHASE12_CATEGORY_ROOT_CAUSE.md").write_text(_category_text(identity), encoding="utf-8")
    (paths.reports_root / "V4_PHASE12_2021_ALIGNMENT.md").write_text(_alignment_text(alignment), encoding="utf-8")
    (paths.reports_root / "V4_PHASE12_IDENTITY_READINESS.md").write_text(_identity_text(identity, panel, holdout), encoding="utf-8")
    (paths.reports_root / "V4_PHASE12_BASELINE_PROTOCOL.md").write_text(_protocol_text(panel), encoding="utf-8")


def _category_text(identity: list[dict[str, object]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in identity:
        counts[str(row["category_change_reason"])] += 1
    lines = [
        "# V4 Phase 12｜2024 類型斷點",
        "",
        "五個 B 級地理群的 2024 類型異常，不是這份程式把上一列類型填進空白儲存格。寬表 CSV 的每一列都有自己的「類型」欄，欄位數是 19，縣市裡的英文換行留在同一個欄位。",
        "",
        "Excel 113 年工作簿的類型欄是合併儲存格，值在合併區左上角，據點那一列的 A 欄是空的。梅山太平雲梯在 Excel 第 106 列，合併範圍的類型是國家級風景特定區，1 月人次 24249，與 CSV 同一列的人次相同。CSV 第 3981 筆卻把類型寫成宗教場所。汶水遊客中心、金水學校、太魯閣相關據點也是 CSV 類型和 Excel 合併類型不同。",
        "",
        "沒有文件記載這些據點被官方改類。類型名稱看起來不合理，也不能因此改寫 CSV。CSV 類型和同年 Excel 合併類型不一致時，原因記為 unresolved。parser_error 只在解析器把合併區以外的類型填進去時才成立，這五群沒有這個情形。CSV 2023 與 2024 類型相同的列記為 unchanged，即使 Excel 用「國家(自然)公園」、CSV 用「國家公園」。",
        "",
        f"稽核列的原因次數：{dict(counts)}。",
        "",
        "| 據點 | CSV 2023 類型 | CSV 2024 類型 | Excel 2024 合併類型 | 原因 | 可安全接續 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in identity:
        lines.append(
            f"| {row['official_name']} | {row['csv_category_2023']} | {row['csv_category_2024']} | {row['excel_category_2024']} | {row['category_change_reason']} | {row['safe_to_link']} |"
        )
    lines.extend(["", "排雲山莊的群組縣市是南投縣。2024 CSV 有一筆嘉義縣的排雲山莊，縣市不同，沒有接上，也沒有補值。圓潭自然生態園區沒有同名的 2024 列。Excel 另有「排雲山莊及玉山群峰步道」「圓潭生態園區」，名稱不同，沒有用模糊比對接上。", ""])
    return "\n".join(lines)


def _alignment_text(alignment: list[dict[str, object]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in alignment:
        counts[str(row["pattern"])] += 1
    focus = [row for row in alignment if row["official_name"] in {"觸口遊客中心", "梅山太平雲梯", "達娜伊谷", "圓潭自然生態園區"}]
    lines = [
        "# V4 Phase 12｜2021 年 11–12 月對齊",
        "",
        "阿里山這幾列的 Excel 人次，就在該據點那一列的 110年11月、110年12月儲存格。月份欄沒有合併。阿里山標題列在第 97 列，縣市和月份是空的，它是遊憩區標題，不是插在據點之間的空白列。",
        "",
        "解析值等於該列儲存格，所以這不是解析器把上一列讀成這一列。CSV 與 Excel 是兩套發布值。兩套都保留，chosen_version 留空。",
        "",
        "相鄰列的數字可以相同。這個相同只描述模式，不能單獨證明正確配對，也沒有用來改寫任何月份。",
        "",
        f"B 類 146 筆的模式次數：{dict(counts)}。parser_error 筆數是 {sum(1 for row in alignment if row['parser_error'])}。",
        "",
        "## 阿里山",
        "",
    ]
    for row in focus:
        lines.append(
            f"- {row['official_name']}：Excel 列 {row['excel_row']}，CSV 11月 {row['csv_november']}、Excel 11月 {row['excel_november']}，上一筆 {row['previous_name']}，模式 {row['pattern']}。"
        )
    lines.extend(
        [
            "",
            "圓潭自然生態園區的 CSV 名稱對不到 Excel 的「圓潭生態園區」，所以維持 excel_name_unmatched。沒有因為數字接近就配對。",
            "",
            "同一張工作簿裡，其他 B 類據點若也出現「CSV 兩個月都等於上一筆 Excel」的模式，次數在上面的 pattern 統計。那仍是來源之間的對照，不是已確認的解析錯誤。",
            "",
        ]
    )
    return "\n".join(lines)


def _identity_text(identity: list[dict[str, object]], panel: list[dict[str, object]], holdout: dict[str, object]) -> str:
    lines = [
        "# V4 Phase 12｜身分與預測資格",
        "",
        "safe_to_link 只在同縣市、同名、CSV 類型沒有改變時為真。跨類型的同名列維持 rule_inferred_not_confirmed，不接進正式集合。沒有人工確認列。",
        "",
        "| 地理群 | 據點數 | 可安全接續 | 未找到 2024 | 類型未解 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in identity:
        grouped[f"{row['county']}／{row['parent_area']}"].append(row)
    for name, rows in grouped.items():
        lines.append(
            f"| {name} | {len(rows)} | {sum(1 for row in rows if row['safe_to_link'])} | {sum(1 for row in rows if row['identity_status'] == 'not_found')} | {sum(1 for row in rows if row['category_change_reason'] == 'unresolved')} |"
        )
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in panel:
        by_group[str(row["group_id"])].add(str(row["year_month"]))
    lines.extend(["", "## 兩種資格", "", "exploratory_retrospective 使用 2024 年以前、Phase 11 已標成可定義份額的月份。發布日未知，strict_asof_eligible 全部為 false。延遲只記成假設，沒有填一個公布日。", ""])
    lines.append(f"候選面板列數 {len(panel)}。confirmatory_holdout_candidate 為 false。")
    lines.append("")
    lines.append("事先指定的探索性回測長度是連續 24 個訓練月加 12 個回測月。窗口數不是獨立樣本數。")
    lines.append("")
    labels = {row["group_id"]: f"{row['county']}／{row['parent_area']}" for row in identity}
    for group_id, months in by_group.items():
        origins = retrospective_origin_count(sorted(months))
        lines.append(f"- {labels.get(group_id, group_id)}：可定義月份 {len(months)}，回溯區段起點 {origins}。")
    lines.extend(
        [
            "",
            f"2025–2026 staging 列數 {holdout['rows']}，年份 {holdout['years']}，used_for_group_selection 為 {holdout['used_for_group_selection']}。人次欄沒有讀入，沒有用來選群或校準。",
            "",
            "2025 可以繼續保持完全不參與模型開發。確認性測試集要等據點、月份、來源優先順序、測量限制和模型選擇都先凍結。這一輪沒有凍結確認性集合。",
            "",
        ]
    )
    return "\n".join(lines)


def _protocol_text(panel: list[dict[str, object]]) -> str:
    vectors = defaultdict(list)
    for row in panel:
        vectors[(row["group_id"], row["year_month"])].append(float(row["share"]))
    invalid = sum(1 for shares in vectors.values() if not valid_share_vector(shares))
    lines = [
        "# V4 Phase 12｜非 LLM 基準規格",
        "",
        "這一輪不擬合模型。目標變數是 reported_visit_distribution，也就是一組據點的官方回報人次份額。它不是個人選擇機率。",
        "",
        "沒有可觀察的歷史社會資訊曝光，就不能把份額持續性說成社會學習的因果效果。WorldLab 的 Q 仍是模型裡有社會資訊與沒有社會資訊的反事實機率距離，不能由這些人次直接計算。",
        "",
        "所有基準使用同一組 site_uid、同一批 exploratory_retrospective 月份，以及相同的預測期限。滾動窗口高度重疊，窗口數不能當成獨立樣本數。",
        "",
        "## 基準",
        "",
        "- 上月份額持續：用 t-1 的份額向量預測 t。t-1 必須是日曆上的前一個月，而且當時已經可定義。",
        "- 去年同月：用 t-12 的份額向量預測 t。",
        "- 歷史同月季節平均：只用 t 之前、同一個日曆月的份額平均。",
        "- 過去 3 個月平均與過去 12 個月平均：只用 t 之前的可定義月份，缺口月份不補 0。",
        "- 組合份額的簡單統計模型：若之後要做，必須在看到 2025 之前寫下規格。這一輪不估計係數。",
        "",
        "## 評分",
        "",
        "主要輸出是完整份額向量。向量必須非負且加總為 1。",
        "",
        "- 份額 MAE",
        "- 份額總變異距離",
        "- 標準化 HHI 的預測誤差",
        "- 領先景點是否預測正確。平手時不把空白領先算成命中。",
        "",
        f"目前候選面板每個群月的份額向量，加總檢查失敗的群月數是 {invalid}。",
        "",
        "## 還沒有消除的測量限制",
        "",
        "計數仍是 B 級 conditional，沒有據點級文件，不能升成 A。2021 年 11–12 月的兩套來源沒有選邊。全群零人次的月份沒有份額。發布日未知，所以沒有 strict_asof_eligible。2024 類型在 CSV 與 Excel 之間未解的列不能進確認性集合。外部身分驗證仍是 unknown。",
        "",
    ]
    return "\n".join(lines)


def _only(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if len(rows) != 1:
        return None
    return rows[0]


def _months_match(csv_row: dict[str, object] | None, excel_row: dict[str, object] | None) -> bool:
    if csv_row is None or excel_row is None:
        return False
    csv_months = dict(csv_row["months"])
    excel_months = dict(excel_row["months"])
    if set(csv_months) != set(range(1, 13)) or set(excel_months) != set(range(1, 13)):
        return False
    return all(abs(float(csv_months[month]) - float(excel_months[month])) <= 1e-6 for month in range(1, 13))


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _hashes() -> dict[str, str]:
    return {relative: file_sha256(WORLD_ROOT / relative) for relative in GUARDED}


if __name__ == "__main__":
    print(json.dumps(run_phase12(), ensure_ascii=False, indent=2))
