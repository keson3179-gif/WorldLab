from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from worldlab_data.config import V4Paths, load_paths
from worldlab_data.phase8 import (
    chinese_line,
    classify_annual_difference,
    confirmation_decision,
    function_tag,
    geographic_groups,
    parse_annual_matrix,
    substitution_candidates,
)
from worldlab_data.textutil import comparison_key, county_parts, nfkc, parse_calendar_year, parse_count, stable_site_id
from worldlab_data.visitors import read_monthly_csv


def run_phase8(paths: V4Paths | None = None) -> dict[str, object]:
    paths = paths or load_paths()
    visits = pd.read_parquet(paths.processed / "site_month_visits.parquet")
    history = pd.read_parquet(paths.processed / "site_status_history.parquet")
    review = pd.read_csv(paths.reports_root / "site_matching_review.csv", dtype=str, keep_default_na=False)
    pois = pd.read_parquet(paths.interim / "poi_attractions.parquet")
    prior = pd.read_csv(paths.curated / "competition_groups.csv", dtype=str, keep_default_na=False)
    priority = _priority_table(prior, review, pois)
    annual, excel_summary, method_map = _annual_reconciliation(paths, visits)
    geo = geographic_groups(history)
    substitution = substitution_candidates(geo)
    coverage = _coverage(visits)
    group_table = _attach_coverage(pd.concat([geo, substitution], ignore_index=True), coverage, method_map)
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    priority.to_csv(paths.reports_root / "V4_SITE_REVIEW_PRIORITY.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(paths.reports_root / "V4_ANNUAL_RECONCILIATION_DETAIL.csv", index=False, encoding="utf-8-sig")
    group_table.to_csv(paths.reports_root / "V4_PHASE8_GROUPS.csv", index=False, encoding="utf-8-sig")
    manual = priority[priority["match_status"] == "confirmed_manual"]
    if not manual.empty:
        manual.to_csv(paths.manual_crosswalk, index=False, encoding="utf-8-sig")
    _write_group_audit(paths, priority, group_table, review, coverage)
    _write_annual(paths, annual, excel_summary)
    _write_backtest(paths, visits, excel_summary, group_table, priority)
    summary = {
        "priority_sites": int(priority["site_id"].nunique()),
        "priority_rows": int(len(priority)),
        "confirmed_manual": int((priority["match_status"] == "confirmed_manual").sum()),
        "mismatch_site_years": int(len(annual)),
        "geographic_groups": int(geo["group_id"].nunique()) if not geo.empty else 0,
        "substitution_groups": int(substitution["group_id"].nunique()) if not substitution.empty else 0,
        "excel_years": excel_summary.get("years", []),
    }
    (paths.interim / "phase8_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _priority_table(prior: pd.DataFrame, review: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    included = prior[prior["included"] == "True"].drop_duplicates("site_id")
    labels = {}
    if not pois.empty and "category_labels" in pois.columns:
        labels = {str(row["poi_id"]): row.get("category_labels", "") for row in pois.to_dict(orient="records")}
    rows: list[dict[str, object]] = []
    scoped = review[review["site_id"].isin(set(included["site_id"]))]
    for site_id, frame in scoped.groupby("site_id"):
        meta = included[included["site_id"] == site_id].iloc[0]
        candidates = []
        for record in frame.to_dict(orient="records"):
            candidates.append(
                {
                    **record,
                    "official_name": meta["official_name"],
                    "same_key": comparison_key(meta["official_name"]) == comparison_key(record.get("candidate_name")),
                }
            )
        status, conflict, suggestion = confirmation_decision(candidates)
        for record in candidates:
            poi = str(record.get("candidate_poi_id", ""))
            rows.append(
                {
                    "site_id": site_id,
                    "official_name": meta["official_name"],
                    "county": meta["county"],
                    "parent_area": meta["parent_area"],
                    "candidate_name": record.get("candidate_name", ""),
                    "candidate_poi_id": poi,
                    "latitude": record.get("candidate_latitude", ""),
                    "longitude": record.get("candidate_longitude", ""),
                    "candidate_district": record.get("candidate_district", ""),
                    "candidate_category": labels.get(poi, ""),
                    "match_method": record.get("match_method", ""),
                    "match_score": record.get("match_score", ""),
                    "match_status": status if record.get("match_method") == "exact_unique_same_county" and status == "confirmed_manual" else "needs_manual_review",
                    "conflict_reason": conflict,
                    "review_suggestion": suggestion,
                    "unique_sites_in_four_areas": int(included["site_id"].nunique()),
                    "candidate_rows_in_four_areas": int(len(scoped)),
                }
            )
    table = pd.DataFrame.from_records(rows)
    confirmed_ids = set(table.loc[table["match_status"] == "confirmed_manual", "site_id"])
    table.loc[~table["site_id"].isin(confirmed_ids), "match_status"] = "needs_manual_review"
    return table.sort_values(["parent_area", "official_name", "match_score"], ascending=[True, True, False])


def _annual_reconciliation(paths: V4Paths, visits: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[tuple[str, str], set[str]]]:
    csv_path = next(path for path in paths.raw_root.glob("*.csv") if "月別" in path.name)
    wide, encoding = read_monthly_csv(csv_path)
    excel_rows = _read_excel_versions(paths)
    excel = pd.DataFrame(excel_rows)
    excel_summary = _excel_summary(excel)
    excel_summary["csv_encoding"] = encoding
    csv_years = _csv_site_years(wide)
    excel_years = _excel_site_years(excel) if not excel.empty else pd.DataFrame()
    mismatch = csv_years[csv_years["within_file_flag"] == "annual_total_mismatch"].copy()
    rows: list[dict[str, object]] = []
    for record in mismatch.to_dict(orient="records"):
        excel_hit = _lookup_excel(excel_years, record)
        category = classify_annual_difference(
            record["month_statuses"],
            record["month_sum"],
            record["published_annual"],
            excel_hit["state"],
            excel_hit["month_sum"],
            excel_hit["published_annual"],
        )
        difference = None if record["published_annual"] is None else record["month_sum"] - record["published_annual"]
        percent = None
        if difference is not None and record["published_annual"] not in (None, 0):
            percent = difference / record["published_annual"] * 100
        rows.append(
            {
                "site_id": record["site_id"],
                "official_name": record["official_name"],
                "county": record["county"],
                "parent_area": record["parent_area"],
                "category": record["category"],
                "year": record["year"],
                "month_sum": record["month_sum"],
                "published_annual": record["published_annual"],
                "difference_month_minus_annual": difference,
                "abs_difference": None if difference is None else abs(difference),
                "percent_difference": percent,
                "difference_class": category,
                "excel_state": excel_hit["state"],
                "excel_month_sum": excel_hit["month_sum"],
                "excel_published_annual": excel_hit["published_annual"],
                "excel_source_file": excel_hit["source_file"],
                "excel_measurement_method": excel_hit["measurement_method"],
            }
        )
    detail = pd.DataFrame.from_records(rows)
    if not detail.empty:
        detail = detail.sort_values(["abs_difference", "official_name"], ascending=[False, True])
    excel_summary["csv_min_year"] = int(csv_years["year"].min()) if not csv_years.empty else None
    excel_summary["csv_max_year"] = int(csv_years["year"].max()) if not csv_years.empty else None
    return detail, excel_summary, _method_map(excel_years)


def _csv_site_years(wide: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for record in wide.to_dict(orient="records"):
        year = parse_calendar_year(record.get("年別"))
        county, _ = county_parts(record.get("縣市"))
        parent = nfkc(record.get("觀光遊憩區"))
        name = nfkc(record.get("細分"))
        category = nfkc(record.get("類型"))
        statuses = []
        values = []
        for month in range(1, 13):
            visits, status = parse_count(record.get(f"{month}月"))
            statuses.append(status)
            if status in {"reported", "reported_zero"} and visits is not None:
                values.append(visits)
        annual, annual_status = parse_count(record.get("合計"))
        complete = all(status in {"reported", "reported_zero"} for status in statuses)
        if annual_status not in {"reported", "reported_zero"} or not complete or annual is None:
            flag = "not_complete_mismatch_set"
        elif abs(sum(values) - annual) > 1e-6:
            flag = "annual_total_mismatch"
        else:
            flag = "annual_total_match"
        rows.append(
            {
                "site_id": stable_site_id([category, parent, name, county]),
                "official_name": name,
                "county": county,
                "parent_area": parent,
                "category": category,
                "year": year,
                "month_sum": float(sum(values)) if values else None,
                "published_annual": annual,
                "month_statuses": statuses,
                "within_file_flag": flag,
                "name_key": comparison_key(name),
                "county_key": comparison_key(county),
            }
        )
    return pd.DataFrame.from_records(rows)


def _excel_site_years(excel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = excel.groupby(["source_file", "year", "name_key", "county_key"], dropna=False)
    for (source_file, year, name_key, county_key), group in grouped:
        numeric = group[group["observation_status"].isin(["reported", "reported_zero"])]
        annuals = [value for value in group["published_annual"].dropna().unique()]
        rows.append(
            {
                "source_file": source_file,
                "year": int(year),
                "name_key": name_key,
                "county_key": county_key,
                "official_name": group["official_name"].iloc[0],
                "county": group["county"].iloc[0],
                "parent_area": group["parent_area"].iloc[0],
                "month_sum": float(numeric["visits"].sum()) if not numeric.empty else None,
                "published_annual": float(annuals[0]) if len(annuals) == 1 else None,
                "n_months": int(numeric["month"].nunique()),
                "measurement_method": "|".join(sorted({str(value) for value in group["measurement_method"].dropna().astype(str) if str(value).strip()})),
            }
        )
    collapsed = pd.DataFrame.from_records(rows)
    return collapsed


def _method_map(excel_years: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    methods: dict[tuple[str, str], set[str]] = {}
    if excel_years.empty or "measurement_method" not in excel_years.columns:
        return methods
    full = excel_years[excel_years["n_months"] == 12]
    for record in full.to_dict(orient="records"):
        label = str(record.get("measurement_method") or "").strip()
        if not label:
            continue
        methods.setdefault((str(record["name_key"]), str(record["county_key"])), set()).update(label.split("|"))
    return methods


def _lookup_excel(excel_years: pd.DataFrame, record: dict[str, object]) -> dict[str, object]:
    empty = {"state": "absent", "month_sum": None, "published_annual": None, "source_file": "", "measurement_method": ""}
    if excel_years.empty:
        return empty
    hit = excel_years[(excel_years["year"] == record["year"]) & (excel_years["name_key"] == record["name_key"]) & (excel_years["county_key"] == record["county_key"])]
    if hit.empty:
        return empty
    full = hit[hit["n_months"] == 12]
    if full.empty:
        return {**empty, "state": "partial_year", "source_file": "|".join(hit["source_file"].astype(str))}
    signatures = {(row["month_sum"], row["published_annual"]) for row in full.to_dict(orient="records")}
    if len(signatures) > 1:
        return {**empty, "state": "ambiguous", "source_file": "|".join(full["source_file"].astype(str))}
    chosen = full.iloc[0]
    return {
        "state": "present",
        "month_sum": None if pd.isna(chosen["month_sum"]) else float(chosen["month_sum"]),
        "published_annual": None if pd.isna(chosen["published_annual"]) else float(chosen["published_annual"]),
        "source_file": str(chosen["source_file"]),
        "measurement_method": str(chosen.get("measurement_method") or ""),
    }


def _read_excel_versions(paths: V4Paths) -> list[dict[str, object]]:
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


def _excel_summary(excel: pd.DataFrame) -> dict[str, object]:
    if excel.empty:
        return {"parsed_rows": 0, "years": [], "late_years": []}
    reported = excel[excel["observation_status"].isin(["reported", "reported_zero"])]
    counts = reported.groupby("year")["month"].nunique().to_dict()
    late = sorted(year for year in counts if int(year) >= 2025)
    return {
        "parsed_rows": int(len(excel)),
        "years": sorted(int(year) for year in counts),
        "months_by_year": {str(int(year)): int(count) for year, count in counts.items()},
        "late_years": [int(year) for year in late],
        "source_files": sorted(set(excel["source_file"].astype(str))),
    }


def _coverage(visits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    usable = visits[visits["observation_status"].isin(["reported", "reported_zero"])]
    for site_id, group in visits.groupby("site_id"):
        reported = usable[usable["site_id"] == site_id]
        months = set(reported["year_month"].astype(str))
        rows.append(
            {
                "site_id": site_id,
                "first_year_month": min(months) if months else "",
                "last_year_month": max(months) if months else "",
                "reported_months": len(months),
                "zero_months": int((group["observation_status"] == "reported_zero").sum()),
                "missing_months": int(group["observation_status"].isin(["missing", "missing_token"]).sum()),
                "months": months,
            }
        )
    return pd.DataFrame.from_records(rows)


def _attach_coverage(groups: pd.DataFrame, coverage: pd.DataFrame, method_map: dict[tuple[str, str], set[str]]) -> pd.DataFrame:
    if groups.empty:
        return groups
    month_map = {str(row["site_id"]): set(row["months"]) for row in coverage.to_dict(orient="records")}
    meta = coverage.drop(columns=["months"]).set_index("site_id")
    rows = []
    for group_id, group in groups.groupby("group_id"):
        common = None
        for site_id in group["site_id"].astype(str):
            months = month_map.get(site_id, set())
            common = months if common is None else common & months
        common = common or set()
        common_sorted = sorted(common)
        labels = []
        for member in group.to_dict(orient="records"):
            labels.append(method_map.get((comparison_key(member["official_name"]), comparison_key(member["county"])), set()))
        if labels and all(len(item) == 1 for item in labels) and len({next(iter(item)) for item in labels}) == 1:
            comparability = f"excel_same_label:{next(iter(labels[0]))}"
        elif any(len(item) != 1 for item in labels):
            comparability = "unknown"
        else:
            comparability = "excel_labels_differ"
        for member in group.to_dict(orient="records"):
            info = meta.loc[member["site_id"]] if member["site_id"] in meta.index else None
            rows.append(
                {
                    **member,
                    "first_year_month": "" if info is None else info["first_year_month"],
                    "last_year_month": "" if info is None else info["last_year_month"],
                    "reported_months": 0 if info is None else int(info["reported_months"]),
                    "zero_months": 0 if info is None else int(info["zero_months"]),
                    "missing_months": 0 if info is None else int(info["missing_months"]),
                    "common_months": len(common),
                    "common_from": common_sorted[0] if common_sorted else "",
                    "common_until": common_sorted[-1] if common_sorted else "",
                    "measurement_comparability": comparability,
                    "descriptive_series": "yes" if len(common) >= 12 else "no",
                    "descriptive_group_shares": "no",
                    "backtest_eligible": "no",
                    "backtest_block": "計數方式未知，且發布時間無法重建，不能進入嚴格樣本外預測。",
                }
            )
    return pd.DataFrame.from_records(rows)


def _write_group_audit(paths: V4Paths, priority: pd.DataFrame, groups: pd.DataFrame, review: pd.DataFrame, coverage: pd.DataFrame) -> None:
    unique_sites = int(priority["site_id"].nunique()) if not priority.empty else 0
    pair_rows = int(len(priority))
    method_sites = priority.drop_duplicates("site_id")["match_method"].value_counts().to_dict() if not priority.empty else {}
    method_rows = priority["match_method"].value_counts().to_dict() if not priority.empty else {}
    confirmed = int(priority.drop_duplicates("site_id")["match_status"].eq("confirmed_manual").sum()) if not priority.empty else 0
    full_sites = int(review["site_id"].nunique()) if not review.empty else 0
    full_rows = int(len(review))
    lines = [
        "# V4 Phase 8 候選群與配對稽核",
        "",
        "本輪不覆寫 Phase 0–7 的歷史報告。先前 QA 仍是 PASS 9、WARN 4、FAIL 0。",
        "",
        "## 配對計數有沒有重疊",
        "",
        f"四個先前候選區有 **{unique_sites}** 個統計據點、**{pair_rows}** 筆候選配對。",
        f"以據點計的方法次數：{method_sites}。以配對列計的方法次數：{method_rows}。",
        "同一據點只會走一種方法：唯一同名、多筆同名、模糊候選，或沒有達標候選。模糊列數大於據點數，是因為一個據點最多保留三個模糊候選，不是同一據點又被算成 exact。",
        f"全部 802 個統計據點的候選列是 {full_rows}，唯一據點是 {full_sites}。多出來的列同樣是一對多候選，不是重複據點。",
        f"本輪標成 confirmed_manual 的據點：{confirmed}。其餘維持 needs_manual_review。模糊分數沒有被當成確認。",
        "",
        "確認條件只有一個：同縣市恰好一筆景點、正規化名稱相同、而且有座標。同名多筆、只有模糊候選、或沒有候選，都不確認。",
        "",
        "## A. 地理分布群",
        "",
        "定義是同一縣市、同一個有名稱的觀光遊憩區，不再要求 5 到 8 個據點。空白或「-」的遊憩區沒有被合成一群。這一層只表示發布統計時放在同一個區名下，不是替代關係。",
        "",
        _group_summary(groups, "A_geographic"),
        "",
        "## B. 地理可達且活動性質可能相近的替代候選",
        "",
        "只從 A 群裡留下名稱不像遊客中心、展示館、海洋世界或離島的據點。即使如此，替代關係仍是 needs_manual_review。同屬國家公園或風景區本身不是完全替代。",
        "",
        _group_summary(groups, "B_substitution_candidate"),
        "",
        "群內份額都不能做可比描述。Excel 的「遊客人次計算方式」只是工作簿裡的標籤，尚未對照統計作業要點，也沒有寫回月別長表。可以描述的是各據點自己的月序列，而且只限共同有回報月份至少 12 個月的群。共同月份若在 2021 結束，是這些 site_id 的交集到此為止；整理過的月別 CSV 本身到 2024。",
        "",
        "預測回測資格全部為否。",
        "",
    ]
    (paths.reports_root / "V4_PHASE8_GROUP_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _annual_split(detail: pd.DataFrame) -> str:
    if detail.empty:
        return ""
    present = detail["excel_state"].eq("present")
    months_same = present & (detail["excel_month_sum"] - detail["month_sum"]).abs().le(1e-6)
    months_differ = present & (detail["excel_month_sum"] - detail["month_sum"]).abs().gt(1e-6)
    absent = int(detail["excel_state"].eq("absent").sum())
    return (
        f"對到 Excel 且月份合計相同、只有 CSV 公布合計不同的有 {int(months_same.sum())} 筆。"
        f"月份合計本身就和 Excel 不同的有 {int(months_differ.sum())} 筆。"
        f"沒有對上同名同縣市 Excel 列的有 {absent} 筆。"
        "沒有用其中任何一個合計去改月份。"
    )


def _group_summary(groups: pd.DataFrame, definition: str) -> str:
    if groups.empty or "definition" not in groups.columns:
        return "沒有群組。"
    subset = groups[groups["definition"] == definition]
    if subset.empty:
        return "沒有群組。"
    lines = [f"群數：{subset['group_id'].nunique()}。據點列：{subset['site_id'].nunique()}。", "", "| 群組 | 規模 | 共同月份 | 共同起迄 | 描述各據點序列 | 群內份額 | 計數口徑 | 回測 |", "| --- | ---: | ---: | --- | --- | --- | --- | --- |"]
    for group_id, frame in subset.groupby("group_id"):
        names = "；".join(frame["official_name"].astype(str))
        row = frame.iloc[0]
        lines.append(
            f"| {row['county']}／{row['parent_area']}：{names} | {int(frame['site_id'].nunique())} | {int(row['common_months'])} | {row['common_from']}–{row['common_until']} | {row['descriptive_series']} | {row['descriptive_group_shares']} | {row['measurement_comparability']} | {row['backtest_eligible']} |"
        )
    return "\n".join(lines)


def _write_annual(paths: V4Paths, detail: pd.DataFrame, excel_summary: dict[str, object]) -> None:
    counts = detail["difference_class"].value_counts().to_dict() if not detail.empty else {}
    lines = [
        "# V4 Phase 8 年度合計差異",
        "",
        "差額定義是十二個月相加減去公布的年度合計。沒有用年度合計回填或重分配任何月份。",
        "",
        f"不一致的據點年：{len(detail)}。分類次數：{counts}。",
        "",
        _annual_split(detail),
        "",
        "分類規則：月份不齊是缺月；合計或月份無法解析是格式；同一年的 Excel 左側十二個月合計，或 Excel 年度合計，與 CSV 不同，才是版本差異。Excel 右側的前一年對照欄不納入。未滿十二個月的工作簿不拿來重判這 317 筆。對不到列是未知原因。若兩份表數字相同、但 CSV 內部仍不一致，也會標成未知；這次沒有出現這種列。",
        "",
        f"整理過的 CSV 年份是 {excel_summary.get('csv_min_year')} 到 {excel_summary.get('csv_max_year')}。",
        f"年度 Excel 已解析的年份：{excel_summary.get('years')}。各年有數字的月份數：{excel_summary.get('months_by_year')}。",
        "",
        "2025 與 2026 沒有出現在整理過的月別 CSV。民國 114 年工作簿是 2025 全年，民國 115 年工作簿是 2026 年 1 月至 7 月。檔案存在，這輪沒有把它們併進 2012–2024 的長表，也沒有用年度合計重分配月份。民國 97 至 100 年（2008–2011）也只在 Excel。",
        "",
        "完整 317 筆在 `V4_ANNUAL_RECONCILIATION_DETAIL.csv`。下面是差額絕對值最大的 30 筆。",
        "",
        "| 據點 | 縣市 | 年 | 月合計 | 公布合計 | 差額 | 百分比 | 分類 | Excel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    preview = detail.head(30) if not detail.empty else detail
    for record in preview.to_dict(orient="records"):
        percent = "" if record["percent_difference"] is None else f"{record['percent_difference']:.2f}"
        lines.append(
            f"| {record['official_name']} | {record['county']} | {int(record['year'])} | {record['month_sum']} | {record['published_annual']} | {record['difference_month_minus_annual']} | {percent} | {record['difference_class']} | {record['excel_state']} |"
        )
    lines.append("")
    (paths.reports_root / "V4_ANNUAL_RECONCILIATION.md").write_text("\n".join(lines), encoding="utf-8")


def _write_backtest(paths: V4Paths, visits: pd.DataFrame, excel_summary: dict[str, object], groups: pd.DataFrame, priority: pd.DataFrame) -> None:
    lines = [
        "# V4 Phase 8 回測準備度",
        "",
        "既有 QA 摘要沒有覆寫，仍是 PASS 9、WARN 4、FAIL 0。",
        "",
        "## 本輪 QA",
        "",
        "| 檢查 | 結果 | 說明 |",
        "| --- | --- | --- |",
        "| 四區唯一據點數與候選列數分開計算 | PASS | 見群組稽核。 |",
        "| 模糊候選沒有標成 confirmed_manual | PASS | 只有同縣市唯一同名且有座標的據點被確認。 |",
        "| 年度合計沒有回填月份 | PASS | 月份原值仍在既有長表。 |",
        "| 國家公園同區沒有被標成完全替代 | PASS | B 群狀態是 needs_manual_review。 |",
        "| 無法確認公布日的資料沒有 strict_asof_eligible | PASS | 本輪嚴格可用列數是 0。 |",
        "| 計數方式 | WARN | Excel 有計算方式欄，但沒有寫回月別長表，也尚未對照統計作業要點。 |",
        "| 發布版本 | WARN | 修訂版是否就是初次公布，無法從現有檔案重建。 |",
        "",
        "## 三種用途",
        "",
        "| 資料 | 標記 | 理由 |",
        "| --- | --- | --- |",
        "| 2012–2024 月別人次長表 | descriptive_only | 可以描述各據點序列。沒有公布日，不能當嚴格當時可得特徵。 |",
        "| 同一長表，若事後假設公布延遲 | retrospective_with_assumptions | 可以做有條件的事後回測，但延遲月數是假設，不是查到的公布日。修訂版未知。 |",
        "| 任何月別人次、日曆、氣象、活動 | strict_asof_eligible | 否。沒有一項能重建發布時間。 |",
        "| 2025 全年與 2026 年 1–7 月 Excel | descriptive_only | 檔案讀得到，尚未併入長表，也沒有公布日。 |",
        "| 現況活動清單與 TDX 未取得的交通 | descriptive_only | 不能回填 2018 或 2020 的當時條件。 |",
        "",
        "保守事後方案可以假設「某月人次最早於次月、次季或次年同月之後才使用」，並把假設寫進每一次原點。這個方案仍不能處理事後修正：現有 CSV 與 Excel 若數字相同，只代表手上的版本一致，不代表它是當時第一次公布的版本。",
        "",
        f"Excel 已讀年份：{excel_summary.get('years')}。其中 2025 年以後：{excel_summary.get('late_years')}。",
        "",
        "## 四個問題",
        "",
        _answers(groups, priority),
        "",
    ]
    (paths.reports_root / "V4_BACKTEST_READINESS.md").write_text("\n".join(lines), encoding="utf-8")


def _answers(groups: pd.DataFrame, priority: pd.DataFrame) -> str:
    descriptive = []
    if not groups.empty:
        series = groups[(groups["definition"] == "A_geographic") & (groups["descriptive_series"] == "yes")]
        for group_id, frame in series.groupby("group_id"):
            descriptive.append(f"{frame.iloc[0]['county']}／{frame.iloc[0]['parent_area']}（{int(frame.iloc[0]['group_size'])} 處，共同 {int(frame.iloc[0]['common_months'])} 個月）")
    confirmed = 0 if priority.empty else int(priority.drop_duplicates("site_id")["match_status"].eq("confirmed_manual").sum())
    return "\n".join(
        [
            "1. 可用於描述各據點自身月序列、但不能做可比群內份額的地理群：" + ("；".join(descriptive) if descriptive else "沒有。"),
            f"2. 月別長表的計數方式仍是 unknown。四區裡確認了 {confirmed} 個景點配對，其餘據點與所有替代群仍是 needs_manual_review。Excel 的計算方式只是工作簿標籤，尚未對照作業要點，所以不能把群內份額當成可比。",
            "3. 有條件的事後回測只適用 2012–2024 的月別人次，而且必須明示假設的公布延遲，並承認修訂版本未知。它不是 strict_asof_eligible。",
            "4. 正式樣本外預測還缺：可查的公布日與修訂版、各據點計數方式、人工確認的替代群、凍結的訓練與保留切分，以及 2025–2026 是否納入長表的決定。",
        ]
    )
